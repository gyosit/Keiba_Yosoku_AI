using CSV
using DataFrames
using DataConvenience
using PyCall
using Random
using Plots
using StatsBase

using Flux, LinearAlgebra
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold
using Flux.Losses: mse, logitcrossentropy, crossentropy
using BSON: @save, @load
using CUDA

function encodemake(x::Any)
    uniques = unique(x)
    return [convert(Array, uniques .== n) for n in x]
end

function transTime(time::AbstractString)::AbstractFloat
    if findfirst(":", time) != nothing
        min_sec_milli = split(time, ":")
        min = min_sec_milli[1]
        sec_milli = split(min_sec_milli[2], ".")
    else
        min = "0"
        sec_milli = split(time, ".")
    end
    
    sec, milli = sec_milli[1], sec_milli[2]
    return parse(Float32, min) * 60.0 + parse(Float32, sec) + parse(Float32, milli) / 10.0
end

function splitDf(df, pct)
    ids = collect(axes(df, 1))
    shuffle!(ids)
    sel = ids .<= nrow(df) .* pct
    return df[sel, :], df[.!sel, :]
end

function trans32(df::DataFrame)::DataFrame
    len = size(df, 2)
    for i in 1:len
#         if (typeof(df[!, i][1]) == Int64) || (typeof(df[!, i][1]) == Bool)
#             df[!, i] = [convert(Int32, v) for v in df[!, i]]
        if (typeof(df[!, i][1]) == Int64) || (typeof(df[!, i][1]) == Bool) || (typeof(df[!, i][1]) == Float64)
            df[!, i] = [convert(Float32, v) for v in df[!, i]]
        end
    end
    return df
end

function makeOneHot(df::DataFrame, key)::DataFrame
#     df_one_hot = onehotbatch(unique(df[col]), df[col])
#     println(df_one_hot)
#     df = hcat(df, df_one_hot)
#     select!(df, Not(col))
    df_onehot = select(df, [Symbol(key) => ByRow(isequal(v)) => Symbol(key, i) for (i, v) in enumerate(unique(df[key]))])
    df_new = hcat(df, df_onehot)
    df_new = select(df_new, Not(key))
    return df_new
end

function pickupPrediction(df::DataFrame, day)
    return df[(df.day .!= day), :], df[(df.day .== day), :]
end

function deepLearning(train, test, predict, target)
    device = gpu
    loss_f = mse
    
    # データセットを作る
    train_x, train_y = Matrix(select(train, Not(target))), Matrix(train[!, target])
    test_x, test_y =  Matrix(select(test, Not(target))), Matrix(test[!, target])
    predict_x, predict_y = Matrix(select(predict, Not(target))), Matrix(predict[!, target])
#     train_loader = DataLoader((train_x|> device, train_y|> device), batchsize=batch_size, shuffle=true)
#     test_loader = DataLoader((test_x|> device, test_y|> device), batchsize=batch_size, shuffle=true)
    
    active_f = leakyrelu
    model = Chain(
        Dense(size(train_x, 2), 32),
        BatchNorm(32, active_f),
        Dense(32, 16),
        AlphaDropout(0.2),
        BatchNorm(16, active_f),
        Dense(16, 8),
        AlphaDropout(0.2),
        BatchNorm(8, active_f),
        Dense(8, size(train_y, 2), active_f)
    )
    model = fmap(cu, model)
    parameters = Flux.params(model)

    # Define optimizer
    η = 1f-2   # Learning rate
    η_decay = 1f-3
    opt = Flux.Optimiser(ADAM(η), InvDecay(η_decay))
   
    function doTrain(epochs, epoch)
        # ミニバッチの作成
        BATCH_SIZE = 32
        ids = shuffle(1:size(train_x, 1))
        x = train_x[ids, :][1:BATCH_SIZE, :]'
        y = train_y[ids, :][1:BATCH_SIZE, :]'|>gpu

        # 学習
        loss(x, y) = loss_f(model(x|>gpu), y)
        grads = gradient(()->loss(x, y), Flux.params(model))
        Flux.Optimise.update!(opt, Flux.params(model), grads)
        
        # ロス算出
        train_loss, test_loss = 0, 0
        if epoch % (epochs/100) == 0
            ŷ = model(x|>gpu)
            train_loss = sqrt(loss_f(ŷ, y))
            ids = shuffle(1:size(test_x, 1))
            x = test_x[ids, :][1:BATCH_SIZE, :]'
            y = test_y[ids, :][1:BATCH_SIZE, :]'|>gpu
            ŷ = model(x|>gpu)
            test_loss = sqrt(loss_f(ŷ, y))
            println("epoch = $epoch")
            println("    train_loss = $train_loss")
            println("    test_loss = $test_loss")
        end
        
        return train_loss, test_loss
    end
    
    function doTest()
        ls = 0
        for i in 1:size(test_x, 1)
            x = test_x[i, :]
            y = test_y[i]
            ŷ = model(x|>gpu)
            ls += loss_f(ŷ, y)
            println(ŷ, y)
        end
        ls /= size(test_x, 1)
        println("    test_loss = $ls")
    end
    
    function doPredict()
        ls = 0
        res = Dict{Int32, Any}()
        for i in 1:size(predict_x, 1)
            x = predict_x[i, :]
            y = predict_y[i, :]|>gpu
            x = reshape(x, size(x, 1), 1)
            ŷ = model(x|>gpu)
            ls += loss_f(ŷ, y)
            res[i] = ŷ
            println(ŷ, y)
        end
        ls /= size(predict_x, 1)
        ls = sqrt(ls)
        println(sort(collect(res), by=x->x[2][1]))
        println("    predict_loss = $ls")
    end

    epochs = 200000
    train_his, test_his = [], []
    for epoch in 1:epochs
        train_loss, test_loss = doTrain(epochs, epoch)
        if train_loss * test_loss != 0
            push!(train_his, train_loss)
            push!(test_his, test_loss)
        end
    end
    
    # doTest()
    doPredict()
    
    pyplot()
    plot(train_his, label="Train", ylim=1:10)
    plot!(test_his, label="Test")
   
end

function main()
    # ファイルの読み込み
    # Train, val
    horse_df = DataFrame(CSV.File("horse_train.csv"))
    race_df = DataFrame(CSV.File("race_train.csv"))
    df = innerjoin(horse_df, race_df, on = :id)
       
    # 時間換算 [秒]
    df.time = [transTime(time) for time in df.time]
    
    # 速さの生成 [m/s]
    df.velocity = df.len ./ df.time
    
    # カテゴリ変数の置換
    mappings = ["kind", "gender", "hair", "weather", "ground", "ground_type", "jockey", "place"]
    for key in mappings
        df = makeOneHot(df, key)
    end
         
    # 平均タイムと平均順位の算出
    ave_velocitys = []
    ave_ranks = []
    for i in 1:size(df, 1)
        name = df[i, "name"]
        day = df[i, "day"]
        same_data = sort(df[(df.name .== name) .& (df.day .< day), :], [:day], rev=true)
        ave_velocity = convert(Float32, mean(same_data[begin:min(3, end), :velocity]))
        ave_rank = mean(same_data[begin:min(3, end), :rank])
        push!(ave_velocitys, isnan(ave_velocity) ? 0 : ave_velocity)
        push!(ave_ranks, isnan(ave_rank) ? 0 : ave_rank)
    end
    df[!, :ave_velocity] = [convert(Float32, v) for v in ave_velocitys]
    df[!, :ave_rank] = [convert(Float32, v) for v in ave_ranks]
           
    # 型変換 (一応)
    df = trans32(df)
       
    # 予測用データセットの抽出
    println(size(df))
    df, df_predict = pickupPrediction(df,"2022/06/26")
    println(size(df))
    println(size(df_predict))
    
    # 不要な列の削除
    drop_list = [:kind, :day, :birthday, :id, :name, :rank, :popularity, :time]
    for drop_c in drop_list
        if string(drop_c) in names(df)
            df = select!(df, Not(drop_c))
            df_predict = select!(df_predict, Not(drop_c))
        end
    end

    train, test = splitDf(df, 0.7)
    deepLearning(train, test, df_predict, ["velocity"])
end

main()
