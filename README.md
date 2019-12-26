# 2019_Data_Science_Bawl
子供のゲームアプリを分析するコンペ

## Data Descriptionから読み取れること
### 予測対象
- 5つの評価軸があり、それぞれのaccuracy groupをあてる。
  - 評価軸
  > - Bird Measurer
  > - Cart Balancer
  > - Cauldron Filler
  > - Chest Sorter
  > - Mushroom Sorter.
  - 成績の良い子は全評価軸で評価が高い可能性がある。
- target: 子供が何回ゲームに挑戦するかを回数からaccuracy groupを予測する.
  - accuracy group
  > - 一度目の挑戦で成功した子: 3
  > - 二度目の挑戦で成功した子: 2
  > - 三度目以降の挑戦で成功した子: 1
  > - 一度も成功しなかった子: 0
  - 成功までの回数をtargetにして、accuracy groupを算出する方法
  - accuracy groupを直接算出する方法
### その他
- install_idはユニークだけど、デバイスを共有している可能性がある。
- train data setでは、評価のないdataも含まれるが、test set では評価のないdataは含まれない。
