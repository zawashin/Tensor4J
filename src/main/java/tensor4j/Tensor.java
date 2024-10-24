package tensor4j;

import java.io.Serializable;

/**
 * @author Shin-Ichiro Serizawa <zawashin@outlook.com>
 */
public class Tensor implements Cloneable, Serializable {
    public static int RANK_MAX = 4;
    protected int rank;
    protected int[] shape;
    protected int length;
    protected int jklMax;
    protected int klMax;
    protected double[] values;

    public Tensor(double value) {
        this();
        values[0] = value;
    }

    public Tensor(double[] values) {
        this(values.length, 1, 1, 1);
        this.values = values.clone();
    }

    public Tensor(double[][] values) {
        this(values.length, values[0].length);
        int n = 0;
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                this.values[n++] = values[i][j];
            }
        }
    }

    public Tensor(double[][][] values) {
        this(values.length, values[0].length, values[0][0].length);
        int n = 0;
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                for (int k = 0; k < shape[2]; k++) {
                    this.values[n++] = values[i][j][k];
                }
            }
        }
    }

    public Tensor(double[][][][] values) {
        this(values.length, values[0].length, values[0][0].length, values[0][0][0].length);
        int n = 0;
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                for (int k = 0; k < shape[2]; k++) {
                    for (int l = 0; l < shape[3]; l++) {
                        this.values[n++] = values[i][j][k][l];
                    }
                }
            }
        }
    }

    public Tensor(Tensor other) {
        rank = other.rank;
        shape[0] = other.shape[0];
        shape[1] = other.shape[1];
        shape[2] = other.shape[2];
        shape[3] = other.shape[3];
        jklMax = other.jklMax;
        klMax = other.klMax;
        length = other.length;
        shape = other.shape.clone();
        values = other.values.clone();
    }

    public Tensor(double[] values, int[] shape) {
        this(shape);
        this.values = values.clone();
    }

    protected Tensor(int... shape) {
        this.shape = new int[RANK_MAX];
        switch (shape.length) {
            case 0:
                this.shape[0] = 1;
                this.shape[1] = 1;
                this.shape[2] = 1;
                this.shape[3] = 1;
                break;
            case 1:
                this.shape[0] = shape[0];
                this.shape[1] = 1;
                this.shape[2] = 1;
                this.shape[3] = 1;
                break;
            case 2:
                this.shape[0] = shape[0];
                this.shape[1] = shape[1];
                this.shape[2] = 1;
                this.shape[3] = 1;
                break;
            case 3:
                this.shape[0] = shape[0];
                this.shape[1] = shape[1];
                this.shape[2] = shape[2];
                this.shape[3] = 1;
                break;
            case 4:
                this.shape[0] = shape[0];
                this.shape[1] = shape[1];
                this.shape[2] = shape[2];
                this.shape[3] = shape[3];
                break;
            default:
                throw new RuntimeException(Utils.ERROR_RANK);
        }
        if (this.shape[0] == 1 && this.shape[1] == 1 && this.shape[2] == 1 && this.shape[3] == 1) {
            rank = 0;
        } else if (this.shape[0] != 1 && this.shape[1] == 1 && this.shape[2] == 1 && this.shape[3] == 1) {
            rank = 1;
        } else if (this.shape[0] != 1 && this.shape[1] != 1 && this.shape[2] == 1 && this.shape[3] == 1) {
            rank = 2;
        } else if (this.shape[0] != 1 && this.shape[1] != 1 && this.shape[2] != 1 && this.shape[3] == 1) {
            rank = 3;
        } else {
            rank = 4;
        }
        length = this.shape[0] * this.shape[1] * this.shape[2] * this.shape[3];
        jklMax = this.shape[1] * this.shape[2] * this.shape[3];
        klMax = this.shape[2] * this.shape[3];
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

    public double[] getValues() {
        return values;
    }

    public int getImax() {
        return shape[0];
    }

    public int getJmax() {
        return shape[1];
    }

    public int getKmax() {
        return shape[2];
    }

    public int getLmax() {
        return shape[3];
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
        return clone;
    }

    public double getValue() {
        return Utils.getValue(this);
    }

    public double getValue(int i) {
        return Utils.getValue(this, i);
    }

    public double getValue(int i, int j) {
        return Utils.getValue(this, i, j);
    }

    public double getValue(int i, int j, int k) {
        return Utils.getValue(this, i, j, k);
    }

    public double getValue(int i, int j, int k, int l) {
        return Utils.getValue(this, i, j, k, l);
    }

    public void setValue(double value) {
        Utils.setValue(this, value);
    }

    public void setValue(int i, double value) {
        Utils.setValue(this, i, value);
    }

    public void setValue(int i, int j, double value) {
        Utils.setValue(this, i, j, value);
    }

    public void setValue(int i, int j, int k, double value) {
        Utils.setValue(this, i, j, k, value);
    }

    public void setValue(int i, int j, int k, int l, double value) {
        Utils.setValue(this, i, j, k, l, value);
    }

    public String toString() {
        return Utils.toString(this);
    }

    public Tensor plus(Tensor t) {
        return Operators.plus(this, t);
    }

    public Tensor plus(double d) {
        return Operators.plus(this, d);
    }

    public void plusAssign(Tensor t) {
        Operators.plusAssign(this, t);
    }

    public void plusAssign(double d) {
        Operators.plusAssign(this, d);
    }

    public Tensor minus(Tensor d) {
        return Operators.minus(this, d);
    }

    public Tensor minus(double t) {
        return Operators.minus(this, t);
    }

    public void minusAssign(Tensor d) {
        Operators.minusAssign(this, d);
    }

    public void minusAssign(double t) {
        Operators.minusAssign(this, t);
    }

    public Tensor times(Tensor t) {
        return Operators.times(this, t);
    }

    public Tensor times(double d) {
        return Operators.times(this, d);
    }

    public void timesAssign(Tensor d) {
        Operators.timesAssign(this, d);
    }

    public void timesAssign(double t) {
        Operators.timesAssign(this, t);
    }

    public Tensor div(Tensor t) {
        return Operators.div(this, t);
    }

    public Tensor div(double d) {
        return Operators.div(this, d);
    }

    public void divAssign(Tensor d) {
        Operators.divAssign(this, d);
    }

    public void divAssign(double t) {
        Operators.divAssign(this, t);
    }

    public Tensor neg() {
        return Operators.neg(this);
    }

    public Tensor cos() {
        return Operators.cos(this);
    }

    public Tensor sin() {
        return Operators.sin(this);
    }

    public Tensor tanh() {
        return Operators.tanh(this);
    }

    public Tensor exp() {
        return Operators.exp(this);
    }

    public Tensor log() {
        return Operators.log(this);
    }

    public Tensor pow(double index) {
        return Operators.pow(this, index);
    }

    public Tensor square() {
        return Operators.square(this);
    }

    public Tensor transpose() {
        return Operators.transpose(this);
    }

    public Tensor dot(Tensor t) {
        return Operators.dot(this, t);
    }

    public Tensor meanSquaredError(Tensor t) {
        return Operators.meanSquaredError(this, t);
    }

    /*
     * 2階のテンソルまでにしたので不要
     */
    // Stub
    public Tensor reshapeSumBackward(Tensor gy, int[] shape, int axis) {
        if (gy != null) {
            throw new RuntimeException(Utils.NOT_IMPLEMENTED);
        }
        Tensor x = gy.clone();
        return new Tensor(x);
    }

    public Tensor reshape(int[] shape) {
        return Utils.reshape(this, shape);
    }

    public Tensor sum() {
        return Utils.sum(this, 2);
    }

    public Tensor sum(Tensor x, int axis) {
        return Utils.sum(this, axis);
    }

    public Tensor broadcastTo(Tensor x, int[] shape) {
        return Utils.broadcastTo(this, shape);
    }

    public Tensor sumTo(int[] shape) {
        return Utils.sumTo(this, shape);
    }

}
