package tensor4j;

import java.util.Arrays;
import java.util.Random;

/**
 * @author Shin-Ichiro Serizawa <zawashin@outlook.com>
 */
public class Utils {
    public static final String[] ERROR_MESSAGE = new String[]{
            "Tensor Order is not 0th.",
            "Tensor Order is not 1st.",
            "Tensor Order is not 2nd.",
            "Tensor Order is not 3rd.",
            "Tensor Order is not 4th.",
            "Over " + Tensor.RANK_MAX + " Order Tensor is not implemented.",
            "Data Length is not correct",
            "Tensor shape is not match",
            "Stub: Not implemented yet"
    };
    public static final String ERROR_RANK = ERROR_MESSAGE[Tensor.RANK_MAX + 1];
    public static final String ERROR_LENGTH = ERROR_MESSAGE[Tensor.RANK_MAX + 2];
    public static final String ERROR_SHAPE = ERROR_MESSAGE[Tensor.RANK_MAX + 3];
    public static final String NOT_IMPLEMENTED = ERROR_MESSAGE[Tensor.RANK_MAX + 4];

    public static Tensor create(int... shape) {
        return create(0.0, shape);
    }

    public static Tensor create(double value, int[] shape) {
        int[] shape_;
        if (shape.length < Tensor.RANK_MAX) {
            shape_ = new int[Tensor.RANK_MAX];
            Arrays.fill(shape_, 1);
            System.arraycopy(shape, 0, shape_, 0, shape.length);
        } else {
            shape_ = shape;
        }
        int length = shape_[0];
        for (int i = 1; i < shape_.length; i++) {
            length *= shape_[i];
        }
        double[] values = new double[length];
        for (int i = 0; i < length; i++) {
            values[i] = value;
        }
        return new Tensor(values, shape_);
    }

    public static Tensor createRandom(int[] shape) {
        return createRandom(1.0, shape);
    }

    public static Tensor createRandom(double value, int[] shape) {
        int[] shape_;
        if (shape.length < Tensor.RANK_MAX) {
            shape_ = new int[Tensor.RANK_MAX];
            Arrays.fill(shape_, 1);
            System.arraycopy(shape, 0, shape_, 0, shape.length);
        } else {
            shape_ = shape;
        }
        int length = shape_[0];
        for (int i = 1; i < shape_.length; i++) {
            length *= shape_[i];
        }
        double[] values = new double[length];
        Random random = new Random(System.currentTimeMillis());
        for (int i = 0; i < length; i++) {
            values[i] = value * random.nextDouble();
        }
        return new Tensor(values, shape_);
    }

    public static Tensor createRandom(double valueMax, double valueMin, int... shape) {
        return createRandom(valueMax - valueMin, shape);
    }

    public static Tensor to2ndOrder(Tensor t) {
        if (t.rank == 1) {
            int[] shape = new int[]{t.getShape(1), t.getShape(0), t.getShape(2), t.getShape(3)};
            double[][] values = new double[][]{t.getValues()};
            return new Tensor(values);
        } else {
            throw new RuntimeException(Utils.ERROR_RANK);
        }
    }

    /*
     * 2階のテンソルまでにしたので不要
     */
    // Stub
    public static Tensor reshapeSumBackward(Tensor gy, int[] shape, int axis, boolean keepDims) {
        if (gy == null) {
            throw new RuntimeException(Utils.NOT_IMPLEMENTED);
        }
        return gy.clone();
    }

    public static Tensor reshape(Tensor t, int[] shape) {
        int length = shape[0];
        for (int i = 1; i < shape.length; i++) {
            length *= shape[i];
        }
        if (t.length != length) {
            throw new RuntimeException(Utils.ERROR_LENGTH);
        }
        double[] values = new double[length];
        switch (t.rank) {
            case 0:
                return t.clone();
            case 1:
                values = t.values.clone();
            case 2:
                int n = 0;
                for (int i = 0; i < shape[0]; i++) {
                    for (int j = 0; j < shape[1]; j++) {
                        values[i * t.jklMax * j * t.klMax] = t.getValues()[n++];
                    }
                }
                break;
            case 3:
            case 4:
            default:
        }
        return new Tensor(values, shape);
    }

    public static Tensor sum(Tensor t) {
        return sum(t, -1);
    }

    public static Tensor sum(Tensor t, int axis, boolean keepdims) {
        return sum(t, -1);
    }

    public static Tensor sum(Tensor t, int axis) {
        if (t.getRank() > 2) {
            throw new RuntimeException(Utils.ERROR_RANK);
        }
        int[] sumShape = new int[Tensor.RANK_MAX];
        if (axis < 0 || t.getRank() < 2) {
            // 総和 axis < 0
            // スカラ・ベクトルは一意にスカラ
            Arrays.fill(sumShape, 1);
        } else if (axis == 0) {
            sumShape[0] = t.getShape()[1];
            sumShape[1] = 1;
            sumShape[2] = 1;
            sumShape[3] = 1;
        } else if (axis == 1) {
            sumShape[0] = t.getShape()[0];
            sumShape[1] = 1;
            sumShape[2] = 1;
            sumShape[3] = 1;
        }
        int length;
        length = sumShape[0];
        for (int i = 1; i < t.length; i++) {
            length *= sumShape[i];
        }
        double[] values = new double[length];
        if (axis < 0 || t.getRank() < 2) {
            for (int i = 0; i < t.getLength(); i++) {
                values[0] += t.getValues()[i];
            }
        } else if (t.rank == 2 && axis == 0) {
            for (int i = 0; i < t.shape[0]; i++) {
                for (int j = 0; j < t.shape[1]; j++) {
                    values[j] += Utils.getValue(t, i, j);
                }
            }
        } else if (t.rank == 2 && axis == 1) {
            for (int i = 0; i < t.shape[0]; i++) {
                for (int j = 0; j < t.shape[1]; j++) {
                    values[i] += Utils.getValue(t, i, j);
                }
            }
        } else {

        }
        return new Tensor(values, sumShape);
    }

    // ChatGPTで修正
    public static Tensor broadcastTo(Tensor t, int[] shape) {
        int length = shape[0];
        for (int i = 1; i < shape.length; i++) {
            length *= shape[i];
        }
        double[] values = new double[length];
        int n = 0;
        int[] xShape = t.getShape();
        int xShape0 = xShape[0];
        int xShape1 = xShape[1];
        int shape0 = shape[0];
        int shape1 = shape[1];

        for (int j = 0; j < shape1; j++) {
            for (int i = 0; i < shape0; i++) {
                int rowIndex = (xShape0 == 1 || xShape0 == shape0) ? Math.min(i, xShape0 - 1) : i % xShape0;
                int colIndex = (xShape1 == 1 || xShape1 == shape1) ? Math.min(j, xShape1 - 1) : j % xShape1;
                values[n++] = t.getValue(rowIndex, colIndex);
            }
        }
        return new Tensor(values, shape);
    }

    // ChatGPTで修正
    public static Tensor sumTo(Tensor t, int[] shape) {
        double[] values = new double[shape[0] * shape[1]];
        int[] xShape = t.getShape();
        int xShape0 = xShape[0];
        int xShape1 = xShape[1];
        int shape0 = shape[0];
        int shape1 = shape[1];

        for (int i = 0; i < xShape[0]; i++) {
            for (int j = 0; j < xShape[0]; j++) {
                int rowIndex = (shape[0] == 1 || shape[0] == xShape[0]) ? Math.min(i, shape[0] - 1) : i % shape[0];
                int colIndex = (shape1 == 1 || shape1 == xShape1) ? Math.min(j, shape1 - 1) : j % shape1;
                values[rowIndex * shape1 + colIndex] += t.getValue(i, j);
            }
        }
        return new Tensor(values, shape);
    }

    // 2つの Tensor を相互的にブロードキャストするサイズを求める
    // ※Tensorは四則演算の際などに自動的にブロードキャストが行われないため明示的にこの関数を利用する
    public static int[] broadcastShape(Tensor t0, Tensor t1) {
        int[] shape = new int[Tensor.RANK_MAX];
        for (int i = 0; i < Tensor.RANK_MAX; i++) {
            shape[i] = Math.max(t0.shape[i], t1.shape[i]);
        }
        return shape;
    }

    public static String toString(Tensor t) {
        StringBuilder buffer = new StringBuilder();
        int n = 0;
        switch (t.rank) {
            case 0:
                buffer.append("[").append(t.values[0]).append("]");
                break;
            case 1:
                buffer.append("[");
                for (int i = 0; i < t.shape[0]; i++) {
                    buffer.append(t.getValue(i));
                    if (i == t.shape[0] - 1) {
                        buffer.append("]");
                    } else {
                        buffer.append(", ");
                    }
                }
                break;
            case 2:
                buffer.append("[");
                for (int i = 0; i < t.shape[0]; i++) {
                    if (i == 0) {
                        buffer.append("[");
                    } else {
                        buffer.append(" [");
                    }
                    for (int j = 0; j < t.shape[1]; j++) {
                        //buffer.append(t.values[n++]);
                        buffer.append(t.getValue(i, j));
                        if (j == t.shape[1] - 1) {
                            buffer.append("]");
                        } else {
                            buffer.append(", ");
                        }
                    }
                    if (i == t.shape[0] - 1) {
                        buffer.append("]");
                    } else {
                        buffer.append(",\n");
                    }
                }
                break;
            case 3:
                buffer.append("[");
                for (int k = 0; k < t.shape[2]; k++) {
                    if (k == 0) {
                        buffer.append("[");
                    } else {
                        buffer.append("  [");
                    }
                    for (int i = 0; i < t.shape[0]; i++) {
                        if (i == 0) {
                            buffer.append("[");
                        } else {
                            buffer.append("  [");
                        }
                        for (int j = 0; j < t.shape[1]; j++) {
                            buffer.append(t.getValue(i, j, k));
                            //buffer.append(t.values[n++]);
                            if (j == t.shape[1] - 1) {
                                buffer.append("]");
                            } else {
                                buffer.append(", ");
                            }
                        }
                        if (i == t.shape[0] - 1) {
                            buffer.append("]");
                        } else {
                            buffer.append(",\n");
                        }
                    }
                    if (k == t.shape[2] - 1) {
                        buffer.append("]");
                    } else {
                        buffer.append(",\n");
                    }
                }
                break;
            case 4:
                buffer.append("[");
                for (int l = 0; l < t.shape[3]; l++) {
                    if (l == 0) {
                        buffer.append("[");
                    } else {
                        buffer.append(" [");
                    }
                    for (int k = 0; k < t.shape[2]; k++) {
                        if (k == 0) {
                            buffer.append("[");
                        } else {
                            buffer.append("  [");
                        }
                        for (int i = 0; i < t.shape[0]; i++) {
                            if (i == 0) {
                                buffer.append("[");
                            } else {
                                buffer.append("   [");
                            }
                            for (int j = 0; j < t.shape[1]; j++) {
                                buffer.append(t.getValue(i, j, k, l));
                                if (j == t.shape[1] - 1) {
                                    buffer.append("]");
                                } else {
                                    buffer.append(", ");
                                }
                            }
                            if (i == t.shape[0] - 1) {
                                buffer.append("]");
                            } else {
                                buffer.append(",\n");
                            }
                        }
                        if (k == t.shape[2] - 1) {
                            buffer.append("]");
                        } else {
                            buffer.append(",\n");
                        }
                    }
                    if (l == t.shape[3] - 1) {
                        buffer.append("]");
                    } else {
                        buffer.append(",\n");
                    }
                }
                break;
            default:
                throw new RuntimeException(ERROR_RANK);
        }
        return buffer.toString();
    }

    public static double getValue(Tensor t) {
        if (t.rank != 0) try {
            throw new Exception();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return t.values[0];
    }

    public static double getValue(Tensor t, int i) {
        if (t.rank != 1) try {
            throw new Exception();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return t.values[i];
    }

    public static double getValue(Tensor t, int i, int j) {
        if (t.rank != 2) try {
            throw new Exception();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        //return t.values[i * t.jklMax + j * t.klMax];
        return t.values[i * t.jklMax + j];
    }

    public static double getValue(Tensor t, int i, int j, int k) {
        if (t.rank != 3) try {
            throw new Exception();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        //return t.values[i * t.jklMax + j * t.klMax + k * t.shape[3]];
        return t.values[i * t.jklMax + j * t.klMax + k];
    }

    public static double getValue(Tensor t, int i, int j, int k, int l) {
        if (t.rank != 4) try {
            throw new Exception();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return t.values[i * t.jklMax + j * t.klMax + k * t.shape[3] + l];
    }

    public static void setValue(Tensor t, double value) {
        if (t.rank != 0) try {
            throw new Exception();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        t.values[0] = value;
    }

    public static void setValue(Tensor t, int i, double value) {
        if (t.rank != 1) try {
            throw new Exception();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        t.values[i] = value;
    }

    public static void setValue(Tensor t, int i, int j, double value) {
        if (t.rank != 2) try {
            throw new Exception();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        t.values[i * t.jklMax + j] = value;
    }

    public static void setValue(Tensor t, int i, int j, int k, double value) {
        if (t.rank != 3) try {
            throw new Exception();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        t.values[i * t.jklMax + j * t.klMax + k] = value;
    }

    public static void setValue(Tensor t, int i, int j, int k, int l, double value) {
        if (t.rank != 4) try {
            throw new Exception();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        t.values[i * t.jklMax + j * t.klMax + k * t.shape[3] + l] = value;
    }


}
