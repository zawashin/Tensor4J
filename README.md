
# Tensor4J
- **Tensor** **for** **J**ava
- 「[ゼロから作るDeep Learning ③ ―フレームワーク編](https://github.com/oreilly-japan/deep-learning-from-scratch-3)
  」のフレームワーク[DezeroをJavaで実装](https://github.com/zawashin/DeZero4j/tree/main)するためにNumPyの代用として実装してみる。

### 方針
- 4階テンソルまで考慮
  - [Deepnetts Community Edition](https://github.com/deepnetts/deepnetts-communityedition)の[Tensorクラス](https://github.com/deepnetts/deepnetts-communityedition/blob/community-visrec/deepnetts-core/src/main/java/deepnetts/util/Tensor.java)に倣う

## 現状
- 四則演算、数学関数を除くメソッドは深層学習で必要な2階までしか実装していない
  - 4階までは、**必要に応じて**対応は可能(なはず)
  - 四則演算、数学関数
    - Operatorsパッケージ
  - その他の操作メソッド
    - Utilパッケージ

## 参考資料
- [Deepnetts Community Edition](https://github.com/deepnetts/deepnetts-communityedition) 
