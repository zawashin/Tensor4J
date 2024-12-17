package tensor4j;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Shin-Ichiro Serizawa <zawashin@outlook.com>
 */
public class Tensor implements Cloneable, Serializable {
    @Serial
    private static final long serialVersionUID = -5871953224191632639L;
    public static int RANK_MAX = 2;
    protected int rank;
    protected int[] shape;
    protected int length;
    protected double[] values;

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
    }

    public Tensor(double[] values, int... shape) {
        if (shape.length > RANK_MAX) {
            throw new RuntimeException(Utils.ERROR_RANK);
        }
        this.rank = shape.length;
        this.shape = shape.clone();
        this.length = Utils.getLength(shape);
        if (values.length != this.length) {
            throw new IllegalArgumentException("Values array length does not match tensor shapes.");
        }
        this.values = values.clone();
    }

    public Tensor(int... shape) {
        this.shape = shape.clone();
        rank = shape.length;
        if (rank > RANK_MAX) {
            throw new RuntimeException(Utils.ERROR_RANK);
        }
        length = Utils.getLength(shape);
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

    public double getValue(int... indices) {
        return Utils.getValue(this, indices);
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

    public void setValue(double value) {
        Utils.setValue(this, value);
    }

    public void setValue(double value, int... indices) {
        Utils.setValue(this, value, indices);
    }

    public void setValue(double value, int i) {
        Utils.setValue(this, value, i);
    }

    public void setValue(double value, int i, int j) {
        Utils.setValue(this, value, i, j);
    }

    public String toString() {
        return Utils.toString(this);
    }

    public Tensor add(Tensor t) {
        return Operators.add(this, t);
    }

    public Tensor add(double d) {
        return Operators.add(this, d);
    }

    public void addAssign(Tensor t) {
        Operators.addAssign(this, t);
    }

    public void addAssign(double d) {
        Operators.addAssign(this, d);
    }

    public Tensor subtract(Tensor d) {
        return Operators.subtract(this, d);
    }

    public Tensor subtract(double t) {
        return Operators.subtract(this, t);
    }

    public void subtractAssign(Tensor d) {
        Operators.subtractAssign(this, d);
    }

    public void subtractAssign(double t) {
        Operators.subtractAssign(this, t);
    }

    public Tensor multiply(Tensor t) {
        return Operators.multiply(this, t);
    }

    public Tensor multiply(double d) {
        return Operators.multiply(this, d);
    }

    public void multiplyAssign(Tensor d) {
        Operators.multiplyAssign(this, d);
    }

    public void multiplyAssign(double t) {
        Operators.multiplyAssign(this, t);
    }

    public Tensor divide(Tensor t) {
        return Operators.divide(this, t);
    }

    public Tensor divide(double d) {
        return Operators.divide(this, d);
    }

    public void divideAssign(Tensor d) {
        Operators.divideAssign(this, d);
    }

    public void divideAssign(double t) {
        Operators.divideAssign(this, t);
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
        return Utils.transpose(this);
    }

    public Tensor dot(Tensor t) {
        return Operators.dot(this, t);
    }

    public Tensor mse(Tensor t) {
        return Operators.mse(this, t);
    }

    public Tensor reshape(int[] shapes) {
        return Utils.reshape(this, shapes);
    }

    public Tensor sum() {
        return Utils.sum(this, -1);
    }

    public Tensor sum(int axis) {
        return Utils.sum(this, axis);
    }

    public Tensor broadcastTo(int[] shapes) {
        return Utils.broadcastTo(this, shapes);
    }

    public Tensor sumTo(int[] shapes) {
        return Utils.sumTo(this, shapes);
    }

}
