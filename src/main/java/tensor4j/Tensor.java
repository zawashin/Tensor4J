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
    protected int[] shapes;
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
        shapes = other.shapes.clone();
        length = other.length;
        values = other.values.clone();
    }

    public Tensor(double[] values, int... shapes) {
        if (shapes.length > RANK_MAX) {
            throw new RuntimeException(Utils.ERROR_RANK);
        }
        this.rank = shapes.length;
        this.shapes = shapes.clone();
        this.length = Utils.getLength(shapes);
        if (values.length != this.length) {
            throw new IllegalArgumentException("Values array length does not match tensor shapes.");
        }
        this.values = values.clone();
    }

    public Tensor(int... shapes) {
        this.shapes = shapes.clone();
        rank = shapes.length;
        switch (rank) {
            case 0:
                length = 1;
                break;
            case 1:
                length = shapes[0];
                break;
            case 2:
                length = shapes[0] * shapes[1];
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
        return shapes;
    }

    public int getShape(int n) {
        return shapes[n];
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
        clone.shapes = this.shapes.clone();
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

        int[] shapes;
        if (!(ndim == 0 || tupledAxis == null)) {
            // Convert negative indices to positive
            int[] actualAxis = new int[tupledAxis.length];
            for (int i = 0; i < tupledAxis.length; i++) {
                actualAxis[i] = tupledAxis[i] >= 0 ? tupledAxis[i] : tupledAxis[i] + ndim;
            }

            // Sort axis indices
            Arrays.sort(actualAxis);

            // Convert gy shapes to list for easier manipulation
            List<Integer> shapeList = new ArrayList<>();
            for (int dim : gy.shapes) {
                shapeList.add(dim);
            }

            // Insert 1's at the appropriate positions
            for (int a : actualAxis) {
                shapeList.add(a, 1);
            }

            // Convert back to array
            shapes = new int[shapeList.size()];
            for (int i = 0; i < shapeList.size(); i++) {
                shapes[i] = shapeList.get(i);
            }
        } else {
            shapes = gy.shapes;
        }

        // Reshape and return
        return gy.reshape(shapes);
    }
}
