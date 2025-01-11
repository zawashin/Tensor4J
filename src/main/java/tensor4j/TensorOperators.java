package tensor4j;

import java.util.Arrays;

/**
 * @author Shin-Ichiro Serizawa <zawashin@outlook.com>
 */
public class TensorOperators {

    public static Tensor add(Tensor t0, Tensor t1) {
        if (!Arrays.equals(t0.shape, t1.shape)) {
            throw new RuntimeException(TensorUtils.ERROR_SHAPE);
        }
        double[] values = new double[t0.length];
        for (int i = 0; i < t0.length; i++) {
            values[i] = t0.values[i] + t1.values[i];
        }
        return new Tensor(values, t0.shape);
    }

    public static Tensor add(Tensor t0, double d) {
        double[] values = new double[t0.length];
        for (int i = 0; i < t0.length; i++) {
            values[i] = t0.values[i] + d;
        }
        return new Tensor(values, t0.shape);
    }

    public static void addAssign(Tensor t0, Tensor t1) {
        if (!Arrays.equals(t0.shape, t1.shape) && (t1.getRank() > 2 && t0.getRank() > 2)) {
            throw new RuntimeException(TensorUtils.ERROR_SHAPE);
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
        if (!Arrays.equals(t0.shape, t1.shape)) {
            throw new RuntimeException(TensorUtils.ERROR_SHAPE);
        }
        double[] values = new double[t0.length];
        for (int i = 0; i < t0.length; i++) {
            values[i] = t0.values[i] - t1.values[i];
        }
        return new Tensor(values, t0.shape);
    }

    public static Tensor subtract(Tensor t0, double d) {
        double[] values = new double[t0.length];
        for (int i = 0; i < t0.length; i++) {
            values[i] = t0.values[i] - d;
        }
        return new Tensor(values, t0.shape);
    }

    public static void subtractAssign(Tensor t0, Tensor t1) {
        if (!Arrays.equals(t0.shape, t1.shape)) {
            System.err.println(Arrays.toString(t0.shape));
            System.err.println(Arrays.toString(t1.shape));
            throw new RuntimeException(TensorUtils.ERROR_SHAPE);
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
        if (!Arrays.equals(t0.shape, t1.shape)) {
            throw new RuntimeException(TensorUtils.ERROR_SHAPE);
        }
        double[] values = new double[t0.length];
        for (int i = 0; i < t0.length; i++) {
            values[i] = t0.values[i] * t1.values[i];
        }
        return new Tensor(values, t0.shape);
    }

    public static Tensor multiply(Tensor t0, double d) {
        double[] values = new double[t0.length];
        for (int i = 0; i < t0.length; i++) {
            values[i] = t0.values[i] * d;
        }
        return new Tensor(values, t0.shape);
    }

    public static void multiplyAssign(Tensor t0, Tensor t1) {
        if (!Arrays.equals(t0.shape, t1.shape) && (t1.getRank() > 2 && t0.getRank() > 2)) {
            throw new RuntimeException(TensorUtils.ERROR_SHAPE);
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
        if (!Arrays.equals(t0.shape, t1.shape)) {
            throw new RuntimeException(TensorUtils.ERROR_SHAPE);
        }
        double[] values = new double[t0.length];
        for (int i = 0; i < t0.length; i++) {
            values[i] = t0.values[i] / t1.values[i];
        }
        return new Tensor(values, t0.shape);
    }

    public static Tensor divide(Tensor t0, double d) {
        double[] values = new double[t0.length];
        for (int i = 0; i < t0.length; i++) {
            values[i] = t0.values[i] / d;
        }
        return new Tensor(values, t0.shape);
    }

    public static void divideAssign(Tensor t0, Tensor t1) {
        if (!Arrays.equals(t0.shape, t1.shape) && (t1.getRank() > 2 && t0.getRank() > 2)) {
            throw new RuntimeException(TensorUtils.ERROR_SHAPE);
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
        return new Tensor(values, t.shape);
    }

    public static Tensor cos(Tensor t) {
        double[] values = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            values[i] = Math.cos(t.values[i]);
        }
        return new Tensor(values, t.shape);
    }

    public static Tensor sin(Tensor t) {
        double[] values = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            values[i] = Math.sin(t.values[i]);
        }
        return new Tensor(values, t.shape);
    }

    public static Tensor tanh(Tensor t) {
        double[] values = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            values[i] = Math.tanh(t.values[i]);
        }
        return new Tensor(values, t.shape);
    }

    public static Tensor exp(Tensor t) {
        double[] values = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            values[i] = Math.exp(t.values[i]);
        }
        return new Tensor(values, t.shape);
    }

    public static Tensor log(Tensor t) {
        double[] values = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            values[i] = Math.log(t.values[i]);
        }
        return new Tensor(values, t.shape);
    }

    public static Tensor pow(Tensor t, double index) {
        double[] values = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            values[i] = Math.pow(t.values[i], index);
        }
        return new Tensor(values, t.shape);
    }

    public static Tensor square(Tensor t) {
        double[] values = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            values[i] = t.values[i] * t.values[i];
        }
        return new Tensor(values, t.shape);
    }

    public static Tensor dot(Tensor t0, Tensor t1) {
        int[] shape = null;
        int length;
        double[] values = null;
        switch (t0.rank) {
            case 0:
                return TensorOperators.multiply(t1, t0.values[0]);
            case 1:
                switch (t1.rank) {
                    case 0:
                        return TensorOperators.multiply(t0, t1.values[0]);
                    case 1:
                        shape = new int[0];
                        values = new double[1];
                        for (int i = 0; i < t0.shape[0]; i++) {
                            values[0] += t0.getValue(i) * t1.getValue(i);
                        }
                        break;
                    case 2:
                        if (t0.shape[0] != t1.shape[0]) {
                            System.err.print(Arrays.toString(t0.getShape()));
                            System.err.print(Arrays.toString(t1.getShape()));
                            throw new RuntimeException(TensorUtils.ERROR_SHAPE);
                        }
                        shape = new int[]{t1.shape[1]};
                        values = new double[t1.shape[1]];

                        // t0とt1の行ごとのドット積を計算
                        for (int i = 0; i < t1.shape[1]; i++) {
                            double value = 0.0;
                            for (int j = 0; j < t0.shape[0]; j++) {
                                value += t0.getValue(j) * t1.getValue(j, i);
                            }
                            values[i] = value;
                        }
                        break;
                    default:
                        throw new RuntimeException(TensorUtils.ERROR_SHAPE);
                }
                break;
            case 2:
                switch (t1.rank) {
                    case 0:
                        return TensorOperators.multiply(t0, t1.values[0]);
                    case 1:
                        if (t0.shape[1] != t1.shape[0]) {
                            System.err.print(Arrays.toString(t0.getShape()));
                            System.err.print(Arrays.toString(t1.getShape()));
                            throw new RuntimeException(TensorUtils.ERROR_SHAPE);
                        }
                        shape = new int[]{t0.shape[0]};
                        values = new double[t0.shape[0]];
                        for (int i = 0; i < t0.shape[0]; i++) {
                            double value = 0.0;
                            for (int j = 0; j < t0.shape[1]; j++) {
                                value += t0.getValue(i, j) * t1.getValue(j);
                            }
                            values[i] = value;
                        }
                        break;
                    case 2:
                        if (t0.shape[1] != t1.shape[0]) {
                            System.err.print(Arrays.toString(t0.getShape()));
                            System.err.print(Arrays.toString(t1.getShape()));
                            throw new RuntimeException("Tensor Shape Error");
                        }
                        length = t0.getShape(0) * t1.getShape(1);
                        shape = new int[]{t0.shape[0], t1.shape[1]};
                        values = new double[length];

                        // 行列積を計算
                        for (int i = 0; i < t0.shape[0]; i++) {
                            for (int j = 0; j < t1.shape[1]; j++) {
                                double value = 0.0;
                                for (int k = 0; k < t0.shape[1]; k++) {
                                    value += t0.getValue(i, k) * t1.getValue(k, j);
                                }
                                values[i * shape[1] + j] = value;
                            }
                        }
                        break;
                    default:
                        throw new RuntimeException(TensorUtils.ERROR_RANK);
                }
                break;
            default:
                throw new RuntimeException(TensorUtils.ERROR_SHAPE);
        }
        return new Tensor(values, shape);
    }

    public static Tensor mse(Tensor t0, Tensor t1) {
        if (t0.length != t1.length) {
            System.err.print(Arrays.toString(t0.shape));
            System.err.print(Arrays.toString(t1.shape));
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
