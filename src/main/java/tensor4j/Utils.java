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
            case 3:
            case 4:
                if (axes.length != 2) {
                    System.err.println("# of Axes must be 0 or 2");
                    throw new RuntimeException("# of Axes must be 0 or 2");
                }
            default:
                throw new RuntimeException(Utils.NOT_IMPLEMENTED);
        }
        return tr;
    }

    public static Tensor reshapeSumBackward(Tensor gy, int[] xshape, boolean keepDims, int... axes) {
        try {
            int ndim = xshape.length;
            int[] tupledAxis;

            // Convert axis to array form
            if (axes == null) {
                tupledAxis = null;
            } else if (axes.length == 1) {
                tupledAxis = new int[]{axes[0]};
            } else if (axes instanceof int[]) {
                tupledAxis = axes;
            } else {
                throw new IllegalArgumentException("Axis must be null, Integer, or int[]");
            }

            int[] shape;
            if (!(ndim == 0 || tupledAxis == null || keepDims)) {
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
        } catch (Exception e) {
            System.err.println(e.getMessage());
            System.exit(1);
        }
        return null;
    }

    public static Tensor reshape(Tensor t, int... shape) {
        int length = shape[0];
        for (int i = 1; i < shape.length; i++) {
            length *= shape[i];
        }
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
        return sum(t, axis, false);
    }

    public static Tensor sum(Tensor t, int axis, boolean keepdims) {
        double[] sums = null;
        try {
            if (t.getRank() > 2) {
                throw new RuntimeException(Utils.NOT_IMPLEMENTED);
            }
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
                case 2:
                    if (axis == 0) {
                        sums = new double[t.getShape(1)];
                        for (int i = 0; i < t.shape[0]; i++) {
                            for (int j = 0; j < t.shape[1]; j++) {
                                sums[j] += Utils.getValue(t, i, j);
                            }
                        }
                    } else if (axis == 1) {
                        sums = new double[t.getShape(0)];
                        for (int i = 0; i < t.shape[0]; i++) {
                            for (int j = 0; j < t.shape[1]; j++) {
                                sums[i] += Utils.getValue(t, i, j);
                            }
                        }
                    } else {
                        throw new RuntimeException("Axis must be Less than 2");
                    }
                    break;
                case 3:
                case 4:
                    throw new RuntimeException(Utils.NOT_IMPLEMENTED);
                default:
                    throw new RuntimeException(Utils.ERROR_RANK);
            }
        } catch (RuntimeException exception) {
            System.err.println(exception.getMessage());
            System.exit(0);
        }
        return new Tensor(sums);
    }

    public static Tensor broadcastTo(Tensor t, int[] shape) {
        // ブロードキャスト可能か確認
        int[] newShape = broadcastShape(t.shape, shape);

        // 新しいテンソルのデータを作成
        double[] newValues = new double[getTotalLength(newShape)];

        // ブロードキャストによる値の再配置
        int[] strides = calculateStrides(t.shape);
        for (int i = 0; i < newValues.length; i++) {
            int[] indices = getIndicesFromLinearIndex(i, newShape);
            int originalIndex = getLinearIndexFromIndices(indices, t.shape, strides);
            newValues[i] = t.values[originalIndex];
        }

        return new Tensor(newValues, newShape);
    }

    private static int getTotalLength(int[] shape) {
        int total = 1;
        for (int dim : shape) {
            total *= dim;
        }
        return total;
    }

    private static int[] calculateStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    private static int[] getIndicesFromLinearIndex(int linearIndex, int[] shape) {
        int[] indices = new int[shape.length];
        for (int i = shape.length - 1; i >= 0; i--) {
            indices[i] = linearIndex % shape[i];
            linearIndex /= shape[i];
        }
        return indices;
    }

    private static int getLinearIndexFromIndices(int[] indices, int[] shape, int[] strides) {
        int linearIndex = 0;
        for (int i = 0; i < shape.length; i++) {
            linearIndex += (indices[i] % shape[i]) * strides[i];
        }
        return linearIndex;
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
        int offsetA = maxLength - shape0.length;
        int offsetB = maxLength - shape1.length;

        // 次元ごとにブロードキャストの判定
        for (int i = maxLength - 1; i >= 0; i--) {
            int dimA = (i - offsetA >= 0) ? shape0[i - offsetA] : 1;
            int dimB = (i - offsetB >= 0) ? shape1[i - offsetB] : 1;

            if (dimA == dimB || dimA == 1 || dimB == 1) {
                // ブロードキャスト可能な次元を選択
                castShape[i] = Math.max(dimA, dimB);
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
            case 3:
            case 4:
                break;
            default:
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

    public static double getValue(Tensor t, int... indeces) {
        if (indeces.length == 0) {
            return t.values[0];
        } else if (indeces.length == 1) {
            return t.values[indeces[0]];
        } else if (indeces.length == 2) {
            return t.values[indeces[0] * t.shape[1] + indeces[1]];
        } else if (indeces.length == 3) {
            return t.values[indeces[0] * t.shape1x2 + indeces[1] * t.shape[2] + indeces[2]];
        } else if (indeces.length == 4) {
            return t.values[indeces[0] * t.shape1x2x3 + indeces[1] * t.shape2x3 + indeces[2] * t.shape[3] + indeces[3]];
        }
        throw new RuntimeException(ERROR_RANK + ": rank is " + t.rank);
    }

    public static void setValue(Tensor t, double value, int... indeces) {
        if (t.rank == 0) {
            t.values[0] = value;
        } else if (t.rank == 1) {
            t.values[indeces[0]] = value;
        } else if (t.rank == 2) {
            t.values[indeces[0] * t.shape[1] + indeces[1]] = value;
        } else if (t.rank == 3) {
            t.values[indeces[0] * t.shape1x2 + indeces[1] * t.shape[2] + indeces[2]] = value;
        } else if (t.rank == 4) {
            t.values[indeces[0] * t.shape1x2x3 + indeces[1] * t.shape2x3 + indeces[2] * t.shape[3] + indeces[3]] = value;
        } else {
            throw new RuntimeException(ERROR_RANK + ": rank is " + t.rank);
        }
    }

}
