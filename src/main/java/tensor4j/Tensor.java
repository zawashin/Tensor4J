package tensor4j;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;

/**
 * @author Shin-Ichiro Serizawa <zawashin@outlook.com>
 */
public class Tensor implements Cloneable, Serializable {

    @Serial
    private static final long serialVersionUID = -5871953224191632639L;
    public static int RANK_MAX = 2;
    protected int rank;
    protected int[] shape;
    protected int[] shape_;
    protected int length;
    protected double[] values;
    private int[] multipliers; // 各次元の積

    public Tensor(double value) {
        this();
        values[0] = value;
    }

    public Tensor(double[] values) {
        this(new int[]{values.length});
        System.arraycopy(values, 0, this.values, 0, values.length);
    }

    public Tensor(double[][] values) {
        this(values.length, values[0].length);
        int n = 0;
        for (double[] value : values) {
            for (int j = 0; j < values[0].length; j++) {
                this.values[n++] = value[j];
            }
        }
    }

    public Tensor(Tensor other) {
        rank = other.rank;
        shape = other.shape.clone();
        length = other.length;
        values = other.values.clone();
        multipliers = other.multipliers.clone();
    }

    public Tensor(double[] values, int... shape) {
        this(shape);
        if (shape.length > RANK_MAX) {
            throw new RuntimeException(TensorUtils.ERROR_RANK);
        }
        if (values.length != this.length) {
            throw new IllegalArgumentException("Values array length does not match tensor shape.");
        }
        this.values = values.clone();
    }

    public Tensor(int... shape) {
        this.shape = shape.clone();
        rank = shape.length;
        if (rank > RANK_MAX) {
            throw new RuntimeException(TensorUtils.ERROR_RANK);
        }
        length = 1;
        shape_ = new int[RANK_MAX];
        Arrays.fill(shape_, 1);
        multipliers = new int[shape.length];

        // インデックス計算用の係数を計算
        for (int i = shape.length - 1; i >= 0; i--) {
            multipliers[i] = length;
            length *= shape[i];
            shape_[i] = shape[i];
        }
        values = new double[length];
    }

    public int getRank() {
        return rank;
    }

    public int getLength() {
        return length;
    }

    public int[] getShape() {
        return shape;
    }

    public int getShape(int n) {
        return shape[n];
    }

    public void setShape(int[] shape) {
        this.shape = shape;
    }

    public double[] getValues() {
        return values;
    }

    public Tensor clone() {
        Tensor clone;
        try {
            clone = (Tensor) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
        clone.values = this.values.clone();
        clone.shape = this.shape.clone();
        clone.multipliers = this.multipliers.clone();
        return clone;
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

    public String toString() {
        return TensorUtils.toString(this);
    }

    public Tensor add(Tensor t) {
        return TensorOperators.add(this, t);
    }

    public Tensor add(double d) {
        return TensorOperators.add(this, d);
    }

    public void addAssign(Tensor t) {
        TensorOperators.addAssign(this, t);
    }

    public void addAssign(double d) {
        TensorOperators.addAssign(this, d);
    }

    public Tensor subtract(Tensor d) {
        return TensorOperators.subtract(this, d);
    }

    public Tensor subtract(double t) {
        return TensorOperators.subtract(this, t);
    }

    public void subtractAssign(Tensor d) {
        TensorOperators.subtractAssign(this, d);
    }

    public void subtractAssign(double t) {
        TensorOperators.subtractAssign(this, t);
    }

    public Tensor multiply(Tensor t) {
        return TensorOperators.multiply(this, t);
    }

    public Tensor multiply(double d) {
        return TensorOperators.multiply(this, d);
    }

    public void multiplyAssign(Tensor d) {
        TensorOperators.multiplyAssign(this, d);
    }

    public void multiplyAssign(double t) {
        TensorOperators.multiplyAssign(this, t);
    }

    public Tensor divide(Tensor t) {
        return TensorOperators.divide(this, t);
    }

    public Tensor divide(double d) {
        return TensorOperators.divide(this, d);
    }

    public void divideAssign(Tensor d) {
        TensorOperators.divideAssign(this, d);
    }

    public void divideAssign(double t) {
        TensorOperators.divideAssign(this, t);
    }

    public Tensor neg() {
        return TensorOperators.neg(this);
    }

    public Tensor cos() {
        return TensorOperators.cos(this);
    }

    public Tensor sin() {
        return TensorOperators.sin(this);
    }

    public Tensor tanh() {
        return TensorOperators.tanh(this);
    }

    public Tensor exp() {
        return TensorOperators.exp(this);
    }

    public Tensor log() {
        return TensorOperators.log(this);
    }

    public Tensor pow(double index) {
        return TensorOperators.pow(this, index);
    }

    public Tensor square() {
        return TensorOperators.square(this);
    }

    public Tensor transpose() {
        return TensorUtils.transpose(this);
    }

    public Tensor dot(Tensor t) {
        return TensorOperators.dot(this, t);
    }

    public Tensor mse(Tensor t) {
        return TensorOperators.mse(this, t);
    }

    public Tensor reshape(int[] shape) {
        return TensorUtils.reshape(this, shape);
    }

    public Tensor sum() {
        return TensorUtils.sum(this, -1);
    }

    public Tensor sum(int axis) {
        return TensorUtils.sum(this, axis);
    }

    public Tensor broadcastTo(int[] shape) {
        return TensorUtils.broadcastTo(this, shape);
    }

    public Tensor sumTo(int[] shape) {
        return TensorUtils.sumTo(this, shape);
    }

}
