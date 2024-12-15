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
        if(shape.length > RANK_MAX) {
            throw new RuntimeException(Utils.ERROR_RANK);
        }
        this.rank = shape.length;
        this.shape = shape.clone();
        this.length = Utils.calcLength(shape);
        if (values.length != this.length) {
            throw new IllegalArgumentException("Values array length does not match tensor shape.");
        }
        this.values = values.clone();
    }

    public Tensor(int... shape) {
        this.shape = shape.clone();
        rank = shape.length;
        switch (rank) {
            case 0:
                length = 1;
                break;
            case 1:
                length = shape[0];
                break;
            case 2:
                length = shape[0] * shape[1];
                break;
            default:
                throw new RuntimeException(Utils.ERROR_RANK);
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
        return Utils.transpose(this);
    }

    public Tensor transpose(int... axes) {
        return Utils.transpose(this, axes);
    }

    public Tensor dot(Tensor t) {
        return Operators.dot(this, t);
    }

    public Tensor mse(Tensor t) {
        return Operators.mse(this, t);
    }

    public Tensor reshape(int[] shape) {
        return Utils.reshape(this, shape);
    }

    public Tensor sum() {
        return Utils.sum(this, -1);
    }

    public Tensor sum(int axis) {
        return Utils.sum(this, axis);
    }


    public Tensor sum(int axis, boolean keepidm) {
        return Utils.sum(this, axis);
    }

    public Tensor broadcastTo(int[] shape) {
        return Utils.broadcastTo(this, shape);
    }

    public Tensor sumTo(int[] shape) {
        return Utils.sumTo(this, shape);
    }

    public Tensor reshapeSumBackward(Tensor gy, int[] xshape, int axis) {
        //    public static Variable reshapeSumBackward(Variable gy, int[] xShape, Object axis, boolean keepdims) {
        int ndim = xshape.length;
        int[] tupledAxis = null;

        // Convert axis to array form
        /*
        if (axis == null) {
            tupledAxis = null;
        } else if (axis instanceof Integer) {
            tupledAxis = new int[]{(Integer) axis};
        } else if (axis instanceof int[]) {
            tupledAxis = (int[]) axis;
        } else {
            throw new IllegalArgumentException("Axis must be null, Integer, or int[]");
        }

         */

        int[] shape;
        if (!(ndim == 0 || tupledAxis == null)) {
            // Convert negative indices to positive
            int[] actualAxis = new int[tupledAxis.length];
            for (int i = 0; i < tupledAxis.length; i++) {
                actualAxis[i] = tupledAxis[i] >= 0 ? tupledAxis[i] : tupledAxis[i] + ndim;
            }

            // Sort axis indices
            Arrays.sort(actualAxis);

            // Convert gy shape to list for easier manipulation
            List<Integer> shapeList = new ArrayList<>();
            for (int dim : gy.shape) {
                shapeList.add(dim);
            }

            // Insert 1's at the appropriate positions
            for (int a : actualAxis) {
                shapeList.add(a, 1);
            }

            // Convert back to array
            shape = new int[shapeList.size()];
            for (int i = 0; i < shapeList.size(); i++) {
                shape[i] = shapeList.get(i);
            }
        } else {
            shape = gy.shape;
        }

        // Reshape and return
        return gy.reshape(shape);
    }
}
