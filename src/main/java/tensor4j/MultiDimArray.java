package tensor4j;

/**
 * @author Shin-Ichiro Serizawa <zawashin@outlook.com>
 */
public class MultiDimArray {
    private double[] values; // フラットな1次元配列
    private int[] shape; // 各次元のサイズ
    private int[] multipliers; // 各次元の積
    private int length; // 全要素数

    public MultiDimArray(int... shape) {
        this.shape = shape.clone();
        length = 1;
        multipliers = new int[this.shape.length];

        // インデックス計算用の係数を計算
        for (int i = this.shape.length - 1; i >= 0; i--) {
            multipliers[i] = length;
            length *= this.shape[i];
        }

        values = new double[length];
    }

    // 任意次元のインデックスを1次元インデックスに変換
    private int index(int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException("次元の数が一致しません。");
        }
        int idx = 0;
        for (int i = 0; i < indices.length; i++) {
            idx += indices[i] * multipliers[i];
        }
        return idx;
    }

    // 値を設定
    public void setValue(double value, int... indices) {
        values[index(indices)] = value;
    }

    // 値を取得
    public double getValue(int... indices) {
        return values[index(indices)];
    }
}
