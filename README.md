
# Tensor4J
- **Tensor** **for** **J**ava
- 「[ゼロから作るDeep Learning ③ ―フレームワーク編](https://github.com/oreilly-japan/deep-learning-from-scratch-3)
  」のフレームワーク[DezeroをJavaで実装](https://github.com/zawashin/DeZero4j/tree/main)するためにNumpyの代用として実装してみる。

### 方針
- 4階テンソルまで考慮
  - [Deepnetts Community Edition](https://github.com/deepnetts/deepnetts-communityedition)の[Tensorクラス](https://github.com/deepnetts/deepnetts-communityedition/blob/community-visrec/deepnetts-core/src/main/java/deepnetts/util/Tensor.java)に倣う

## 現状
  - 四則演算、数学関数以外の演算メソッドは2階テンソルまでしか対応していない
    - 四則演算、数学関数
      - Operatorsパッケージ
    - その他の操作メソッド
      - Utilパッケージ
        - 作りかけ

## 参考資料
- [Deepnetts Community Edition](https://github.com/deepnetts/deepnetts-communityedition) 
