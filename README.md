(編集中)
# Keiba_Yosoku_AI
Juliaを用いた競馬予測AIです。  
お遊びで作ったので精度や動作は**全く**保証しません。  
汎用性も拡張性も皆無です。  
Juliaのディープラーニングってこう書くんだなーくらいの参考にしてください。

# 使い方
## CSVファイル
競走馬の共通データであるhorse_train.csvとレース結果であるrace_train.csvを準備します。  
フォーマットのサンプルとして2行分のデータが入ったファイルをリポジトリに入れておきました。  
概ねヘッダを見れば意味はわかると思いますが、`id`はhttps://db.netkeiba.com/horse/xxxxxxxxxx の`xxxxxxxxxx`に入る数字です。  

race_train.csvには予測したいレースのデータも含めてください。  
当然順位などはわかりませんが、ここは適当な数字でOKです。  
ただし、予測したいレースと同日の他のレースデータが入らないようにしてください。

## 実行方法
事前に`using`で使用するライブラリを`Pkg.add`で追加しておいてください。  
動かすだけならば基本的にソースコードを変える必要はありませんが、
```julia
# 予測用データセットの抽出
println(size(df))
df, df_predict = pickupPrediction(df,"2022/06/26")
println(size(df))
println(size(df_predict))
```
の`"2022/06/26"`は必要に応じて変更する必要があります。  
race_train.csvに追加しておいた、予測したいレースのダミーデータの日付に書き換えてください。

# 宝塚記念(2022/6/26)の予測動画


# 連絡先
[@Mr_isoy](https://twitter.com/Mr_isoy)
