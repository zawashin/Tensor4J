package tensor4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * @author Shin-Ichiro Serizawa <zawashin@outlook.com>
 */
public class Utils {
    public static final String[] ERROR_MESSAGE = new String[]{
            "Tensor Order is not 0th.",
            "Tensor Order is not 1st.",
            "Tensor Order is not 2nd.",
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
        return new Tensor(shape);
    }

    public static Tensor fill(double value, int... shape) {
        Tensor t = new Tensor(shape);
        Arrays.fill(t.values, value);
        return t;
    }

    public static Tensor createRandom(int... shape) {
        return createRandom(1.0, shape);
    }

    public static Tensor createRandom(double value, int... shape) {
        Tensor t = new Tensor(shape);
        Random random = new Random(System.currentTimeMillis());
        for (int i = 0; i < t.length; i++) {
            t.values[i] = value * random.nextDouble();
        }
        return t;
    }

    public static Tensor createRandom(double valueMax, double valueMin, int... shape) {
        return createRandom(valueMax - valueMin, shape);
    }

    public static Tensor transpose(Tensor t, int... axes) {
        Tensor tr = null;
        switch (t.rank) {
            case 0:
            case 1:
                // 数学的には存在しない
                // NumPyに合わせてcloneを返す
                tr = t.clone();
                break;
            case 2:
                if (axes.length != 0) {
                    throw new RuntimeException();
                }
                tr = Utils.create(t.shape[1], t.shape[0]);
                for (int i = 0; i < t.shape[0]; i++) {
                    for (int j = 0; j < t.shape[1]; j++) {
                        tr.setValue(t.getValue(i, j), j, i);
                    }
                }
                break;
            default:
                throw new RuntimeException(Utils.ERROR_RANK);
        }
        return tr;
    }

    public static Tensor reshapeSumBackward(Tensor gy, int[] xshape, int... axis) {
         // Reshape and return
         return gy.reshape(xshape);
    }

    public static Tensor reshape(Tensor t, int... shape) {
        int length = calcLength(shape);
        if (t.length != length) {
            throw new RuntimeException(Utils.ERROR_LENGTH);
        }
        Tensor trs = new Tensor(shape);
        trs.values = t.values.clone();
        return trs;
    }

    public static Tensor sum(Tensor t) {
        return sum(t, -1);
    }

    public static Tensor sum(Tensor t, int axis) {
        double[] sums = null;
            if (axis < 0) {
                double sum = 0.0;
                for (int i = 0; i < t.getLength(); i++) {
                    sum += t.getValues()[i];
                }
                return new Tensor(sum);
            }
            switch (t.getRank()) {
                case 0:
                case 1:
                    sums = new double[1];
                    for (int i = 0; i < t.getLength(); i++) {
                        sums[0] += t.getValues()[i];
                    }
                    break;
                default:
                    throw new RuntimeException(Utils.ERROR_RANK);
            }
        return new Tensor(sums);
    }

    public static Tensor broadcastTo(Tensor t, int[] shape) {
    return null;
    }


    public static int calcLength(int... shape) {
        int length = 1;
        for (int j : shape) {
            length *= j;
        }
        return length;
    }

    public static int[] getIndices(int[] shape, int index) {
        int[] indices = new int[shape.length];
        for(int i = 0; i < shape.length; i++) {
            switch(i) {
                case 0:
                    indices[0] = index / shape[1];
                    break;
                case 1:
                    indices[1] = index % shape[1];
                    break;
                default:
                    throw new RuntimeException(ERROR_RANK);
            }
        }
        return indices;
    }

    public static int getIndex(int[] shape, int[] indices) {
        switch(shape.length) {
            case 0:
                return 0;
            case 1:
                return indices[0];
            case 2:
                return indices[0] * shape[1] + indices[1];
            default:
                throw new RuntimeException(ERROR_RANK);
        }
    }

    // 2つの Tensor を相互的にブロードキャストするサイズを求める
    // ※Tensorは四則演算の際などに自動的にブロードキャストが行われないため明示的にこの関数を利用する
    public static int[] broadcastShape(Tensor t0, Tensor t1) {
        return broadcastShape(t0.shape, t1.shape);
    }

    public static int[] broadcastShape(int[] shape0, int[] shape1) {
        // 最大の次元数を取得
        int maxLength = Math.max(shape0.length, shape1.length);

        // 結果の形状を格納する配列
        int[] castShape = new int[maxLength];

        // shapeA と shapeB を後ろ合わせにするためのオフセットを計算
        int offset0 = maxLength - shape0.length;
        int offset1 = maxLength - shape1.length;

        // 次元ごとにブロードキャストの判定
        for (int i = maxLength - 1; i >= 0; i--) {
            int dim0 = (i - offset0 >= 0) ? shape0[i - offset0] : 1;
            int dim1 = (i - offset1 >= 0) ? shape1[i - offset1] : 1;

            if (dim0 == dim1 || dim0 == 1 || dim1 == 1) {
                // ブロードキャスト可能な次元を選択
                castShape[i] = Math.max(dim0, dim1);
            } else {
                // ブロードキャスト不可能な場合
                throw new RuntimeException("Shapes " + Arrays.toString(shape0) +
                        " and " + Arrays.toString(shape1) +
                        " are not broadcastable.");
            }
        }

        return castShape;
    }


    // ChatGPTで修正
    public static Tensor sumTo(Tensor t, int[] shape) {
        double[] values = new double[shape[0] * shape[1]];
        int[] xShape = t.getShape();

        switch (t.rank) {
            case 0:
            case 1:
                break;
            case 2:
                for (int i = 0; i < xShape[0]; i++) {
                    for (int j = 0; j < xShape[0]; j++) {
                        int i_ = (shape[0] == 1 || shape[0] == xShape[0]) ? Math.min(i, shape[0] - 1) : i % shape[0];
                        int j_ = (shape[1] == 1 || shape[1] == xShape[1]) ? Math.min(j, shape[1] - 1) : j % shape[1];
                        values[i_ * shape[1] + j_] += t.getValue(i, j);
                    }
                }
                break;
            default:
                throw  new RuntimeException(ERROR_RANK);
        }
        return new Tensor(values, shape);
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
                for (int i = 0; i < t.values.length; i++) {
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
            default:
                throw new RuntimeException(ERROR_RANK);
        }
        return buffer.toString();
    }

    public static double getValue(Tensor t, int... indices) {
        if (indices.length == 0) {
            return t.values[0];
        } else if (indices.length == 1) {
            return t.values[indices[0]];
        } else if (indices.length == 2) {
            return t.values[indices[0] * t.shape[1] + indices[1]];
        } else {
            throw new RuntimeException(ERROR_RANK + ": rank is " + t.rank);
        }
    }

    public static void setValue(Tensor t, double value, int... indices) {
        if (t.rank == 0) {
            t.values[0] = value;
        } else if (t.rank == 1) {
            t.values[indices[0]] = value;
        } else if (t.rank == 2) {
            t.values[indices[0] * t.shape[1] + indices[1]] = value;
        } else {
            throw new RuntimeException(ERROR_RANK + ": rank is " + t.rank);
        }
    }

}
