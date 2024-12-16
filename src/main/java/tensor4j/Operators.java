package tensor4j;

import java.util.Arrays;

/**
 * @author Shin-Ichiro Serizawa <zawashin@outlook.com>
 */
public class Operators {

    public static Tensor add(Tensor t0, Tensor t1) {
        if (!Arrays.equals(t0.shapes, t1.shapes)) {
            throw new RuntimeException(Utils.ERROR_SHAPE);
        }
        double[] values = new double[t0.length];
        for (int i = 0; i < t0.length; i++) {
            values[i] = t0.values[i] + t1.values[i];
        }
        return new Tensor(values, t0.shapes);
    }

    public static Tensor add(Tensor t0, double d) {
        double[] values = new double[t0.length];
        for (int i = 0; i < t0.length; i++) {
            values[i] = t0.values[i] + d;
        }
        return new Tensor(values, t0.shapes);
    }

    public static void addAssign(Tensor t0, Tensor t1) {
        if (!Arrays.equals(t0.shapes, t1.shapes)) {
            throw new RuntimeException(Utils.ERROR_SHAPE);
        }
        for (int i = 0; i < t0.length; i++) {
            t0.values[i] += t1.values[i];
        }
    }

    public static void addAssign(Tensor t0, double d) {
        for (int i = 0; i < t0.length; i++) {
            t0.values[i] += d;
        }
    }

    public static Tensor subtract(Tensor t0, Tensor t1) {
        if (!Arrays.equals(t0.shapes, t1.shapes)) {
            throw new RuntimeException(Utils.ERROR_SHAPE);
        }
        double[] values = new double[t0.length];
        for (int i = 0; i < t0.length; i++) {
            values[i] = t0.values[i] - t1.values[i];
        }
        return new Tensor(values, t0.shapes);
    }

    public static Tensor subtract(Tensor t0, double d) {
        double[] values = new double[t0.length];
        for (int i = 0; i < t0.length; i++) {
            values[i] = t0.values[i] - d;
        }
        return new Tensor(values, t0.shapes);
    }

    public static void subtractAssign(Tensor t0, Tensor t1) {
        if (!Arrays.equals(t0.shapes, t1.shapes)) {
            throw new RuntimeException(Utils.ERROR_SHAPE);
        }
        for (int i = 0; i < t0.length; i++) {
            t0.values[i] -= t1.values[i];
        }
    }

    public static void subtractAssign(Tensor t0, double d) {
        double[] x0 = t0.values;
        for (int i = 0; i < x0.length; i++) {
            t0.values[i] -= d;
        }
    }

    public static Tensor multiply(Tensor t0, Tensor t1) {
        if (!Arrays.equals(t0.shapes, t1.shapes)) {
            throw new RuntimeException(Utils.ERROR_SHAPE);
        }
        double[] values = new double[t0.length];
        for (int i = 0; i < t0.length; i++) {
            values[i] = t0.values[i] * t1.values[i];
        }
        return new Tensor(values, t0.shapes);
    }

    public static Tensor multiply(Tensor t0, double d) {
        double[] values = new double[t0.length];
        for (int i = 0; i < t0.length; i++) {
            values[i] = t0.values[i] * d;
        }
        return new Tensor(values, t0.shapes);
    }

    public static void multiplyAssign(Tensor t0, Tensor t1) {
        if (!Arrays.equals(t0.shapes, t1.shapes)) {
            throw new RuntimeException(Utils.ERROR_SHAPE);
        }
        for (int i = 0; i < t0.length; i++) {
            t0.values[i] *= t1.values[i];
        }
    }

    public static void multiplyAssign(Tensor t0, double d) {
        for (int i = 0; i < t0.length; i++) {
            t0.values[i] *= d;
        }
    }

    public static Tensor divide(Tensor t0, Tensor t1) {
        if (!Arrays.equals(t0.shapes, t1.shapes)) {
            throw new RuntimeException(Utils.ERROR_SHAPE);
        }
        double[] values = new double[t0.length];
        for (int i = 0; i < t0.length; i++) {
            values[i] = t0.values[i] / t1.values[i];
        }
        return new Tensor(values, t0.shapes);
    }

    public static Tensor divide(Tensor t0, double d) {
        double[] values = new double[t0.length];
        for (int i = 0; i < t0.length; i++) {
            values[i] = t0.values[i] / d;
        }
        return new Tensor(values, t0.shapes);
    }

    public static void divideAssign(Tensor t0, Tensor t1) {
        if (!Arrays.equals(t0.shapes, t1.shapes)) {
            throw new RuntimeException(Utils.ERROR_SHAPE);
        }
        for (int i = 0; i < t0.length; i++) {
            t0.values[i] /= t1.values[i];
        }
    }

    public static void divideAssign(Tensor t0, double d) {
        for (int i = 0; i < t0.length; i++) {
            t0.values[i] /= d;
        }
    }

    public static Tensor neg(Tensor t) {
        double[] values = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            values[i] = -t.values[i];
        }
        return new Tensor(values, t.shapes);
    }

    public static Tensor cos(Tensor t) {
        double[] values = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            values[i] = Math.cos(t.values[i]);
        }
        return new Tensor(values, t.shapes);
    }

    public static Tensor sin(Tensor t) {
        double[] values = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            values[i] = Math.sin(t.values[i]);
        }
        return new Tensor(values, t.shapes);
    }

    public static Tensor tanh(Tensor t) {
        double[] values = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            values[i] = Math.tanh(t.values[i]);
        }
        return new Tensor(values, t.shapes);
    }

    public static Tensor exp(Tensor t) {
        double[] values = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            values[i] = Math.exp(t.values[i]);
        }
        return new Tensor(values, t.shapes);
    }

    public static Tensor log(Tensor t) {
        double[] values = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            values[i] = Math.log(t.values[i]);
        }
        return new Tensor(values, t.shapes);
    }

    public static Tensor pow(Tensor t, double index) {
        double[] values = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            values[i] = Math.pow(t.values[i], index);
        }
        return new Tensor(values, t.shapes);
    }

    public static Tensor square(Tensor t) {
        double[] values = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            values[i] = t.values[i] * t.values[i];
        }
        return new Tensor(values, t.shapes);
    }

    public static Tensor dot(Tensor t0, Tensor t1) {
        int[] shapes = null;
        int length;
        double[] values = null;
        switch (t0.rank) {
            case 0:
                return Operators.multiply(t1, t0.values[0]);
            case 1:
                switch (t1.rank) {
                    case 0:
                        return Operators.multiply(t0, t1.values[0]);
                    case 1:
                        shapes = new int[0];
                        values = new double[1];
                        for (int i = 0; i < t0.shapes[0]; i++) {
                            values[0] += t0.getValue(i) * t1.getValue(i);
                        }
                        break;
                    case 2:
                        if (t0.shapes[0] != t1.shapes[0]) {
                            System.out.print(Arrays.toString(t0.getShapes()));
                            System.out.print(Arrays.toString(t1.getShapes()));
                            throw new RuntimeException(Utils.ERROR_SHAPE);
                        }
                        shapes = new int[]{t1.shapes[1]};
                        values = new double[t1.shapes[1]];

                        // t0とt1の行ごとのドット積を計算
                        for (int i = 0; i < t1.shapes[1]; i++) {
                            double value = 0.0;
                            for (int j = 0; j < t0.shapes[0]; j++) {
                                value += t0.getValue(j) * t1.getValue(j, i);
                            }
                            values[i] = value;
                        }
                        break;
                    default:
                        throw new RuntimeException(Utils.ERROR_SHAPE);
                }
                break;
            case 2:
                switch (t1.rank) {
                    case 0:
                        return Operators.multiply(t0, t1.values[0]);
                    case 1:
                        if (t0.shapes[1] != t1.shapes[0]) {
                            System.out.print(Arrays.toString(t0.getShapes()));
                            System.out.print(Arrays.toString(t1.getShapes()));
                            throw new RuntimeException(Utils.ERROR_SHAPE);
                        }
                        shapes = new int[]{t0.shapes[0]};
                        values = new double[t0.shapes[0]];
                        for (int i = 0; i < t0.shapes[0]; i++) {
                            double value = 0.0;
                            for (int j = 0; j < t0.shapes[1]; j++) {
                                value += t0.getValue(i, j) * t1.getValue(j);
                            }
                            values[i] = value;
                        }
                        break;
                    case 2:
                        if (t0.shapes[1] != t1.shapes[0]) {
                            System.out.print(Arrays.toString(t0.getShapes()));
                            System.out.print(Arrays.toString(t1.getShapes()));
                            throw new RuntimeException("Tensor Shape Error");
                        }
                        length = t0.getShape(0) * t1.getShape(1);
                        shapes = new int[]{t0.shapes[0], t1.shapes[1]};
                        values = new double[length];

                        // 行列積を計算
                        for (int i = 0; i < t0.shapes[0]; i++) {
                            for (int j = 0; j < t1.shapes[1]; j++) {
                                double value = 0.0;
                                for (int k = 0; k < t0.shapes[1]; k++) {
                                    value += t0.getValue(i, k) * t1.getValue(k, j);
                                }
                                values[i * shapes[1] + j] = value;
                            }
                        }
                        break;
                    default:
                        throw new RuntimeException(Utils.ERROR_RANK);
                }
                break;
            default:
                throw new RuntimeException(Utils.ERROR_SHAPE);
        }
        return new Tensor(values, shapes);
    }

    public static Tensor mse(Tensor t0, Tensor t1) {
        if (t0.length != t1.length) {
            System.out.print(Arrays.toString(t0.shapes));
            System.out.print(Arrays.toString(t1.shapes));
            throw new RuntimeException("Tensor Shape Error");
        }
        int length = t0.length;
        double value = 0.0;
        for (int i = 0; i < length; i++) {
            double dx = t0.values[i] - t1.values[i];
            value += dx * dx;
        }
        value /= length;
        return new Tensor(value);
    }

}
