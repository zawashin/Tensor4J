package tensor4j;

import java.util.Arrays;
import java.util.Random;

/**
 * @author Shin-Ichiro Serizawa <zawashin@outlook.com>
 */
public class Utils1 {
    public static final String[] ERROR_MESSAGES = new String[]{
            "Tensor Order is not 0th.",
            "Tensor Order is not 1st.",
            "Tensor Order is not 2nd.",
            "Over " + Tensor.RANK_MAX + " Order Tensor is not implemented.",
            "Data Length is not correct",
            "Tensor shape is not match",
    };
    public static final String ERROR_RANK = ERROR_MESSAGES[Tensor.RANK_MAX + 1];
    public static final String ERROR_LENGTH = ERROR_MESSAGES[Tensor.RANK_MAX + 2];
    public static final String ERROR_SHAPE = ERROR_MESSAGES[Tensor.RANK_MAX + 3];

    public static Tensor create(int... shape) {
        return new Tensor(shape);
    }

    public static Tensor fill(double value, int... shape) {
        Tensor t = new Tensor(shape);
        Arrays.fill(t.values, value);
        return t;
    }

    public static Tensor random(int... shape) {
        return random(1.0, shape);
    }

    public static Tensor random(double value, int... shape) {
        Tensor t = new Tensor(shape);
        Random random = new Random(System.currentTimeMillis());
        for (int i = 0; i < t.length; i++) {
            t.values[i] = value * random.nextDouble();
        }
        return t;
    }

    public static Tensor random(double valueMax, double valueMin, int... shape) {
        return random(valueMax - valueMin, shape);
    }

    public static Tensor transpose(Tensor t) {
        Tensor tr = null;
        switch (t.rank) {
            case 0:
            case 1:
                // 数学的には存在しない
                // NumPyに合わせてcloneを返す
                tr = t.clone();
                break;
            case 2:
                tr = Utils1.create(t.shape[1], t.shape[0]);
                for (int i = 0; i < t.shape[0]; i++) {
                    for (int j = 0; j < t.shape[1]; j++) {
                        tr.setValue(t.getValue(i, j), j, i);
                    }
                }
                break;
            default:
                throw new RuntimeException(Utils1.ERROR_RANK);
        }
        return tr;
    }


    public static Tensor reshape(Tensor t, int... shape) {
        int length = getLength(shape);
        if (t.length != length) {
            throw new RuntimeException(Utils1.ERROR_LENGTH);
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
        if (t.rank == 0) {
            return t.clone();
        } else if (t.rank == 1) {
            sums = new double[1];
            for (int i = 0; i < t.getLength(); i++) {
                sums[0] += t.getValues()[i];
            }
            return new Tensor(sums[0]);
        } else if (t.rank == 2) {
            if (axis == 0) {
                sums = new double[t.shape[1]];
                for (int i = 0; i < t.shape[0]; i++) {
                    for (int j = 0; j < t.shape[1]; j++) {
                        sums[j] += t.getValue(i, j);
                    }
                }
            } else if (axis == 1) {
                sums = new double[t.shape[0]];
                for (int i = 0; i < t.shape[0]; i++) {
                    for (int j = 0; j < t.shape[1]; j++) {
                        sums[i] += t.getValue(i, j);
                    }
                }
            } else {
                double sum = 0.0;
                for (int i = 0; i < t.getLength(); i++) {
                    sum += t.getValues()[i];
                }
                return new Tensor(sum);
            }
        }
        return new Tensor(sums);
    }

    public static Tensor broadcastTo(Tensor t, int[] shape) {
        return broadcastTo(t, shape, false);
    }

    public static Tensor broadcastTo(Tensor t, int[] shape, boolean validate) {
        int[] shape_ = shape.clone();
        if (validate) {
            shape_ = broadcastShape(t.shape, shape);
            if (shape_ == null) {
                return null;
            }
        }
        int length = Utils1.getLength(shape);
        double[] values = new double[length];
        int[] xShape = t.getShape();
        int xShape0 = 1;
        int xShape1 = 1;
        int shape0 = 1;
        int shape1 = 1;
        switch (t.rank) {
            case 0:
                break;
            case 1:
                xShape1 = xShape[0];
                break;
            case 2:
                xShape0 = xShape[0];
                xShape1 = xShape[1];
                break;
            default:
                System.err.println(ERROR_RANK);
                throw new RuntimeException(ERROR_RANK);
        }
        switch (shape.length) {
            case 0:
                break;
            case 1:
                shape1 = shape_[0];
                break;
            case 2:
                shape0 = shape_[0];
                shape1 = shape_[1];
                break;
            default:
                System.err.println(ERROR_RANK);
                throw new RuntimeException(ERROR_RANK);
        }

        int n = 0;
        for (int i = 0; i < shape0; i++) {
            for (int j = 0; j < shape1; j++) {
                int i_ = i % xShape0;
                int j_ = j % xShape1;
                values[n++] = t.getValues()[i_ * xShape1 + j_];
            }
        }
        return new Tensor(values, shape);
    }

    // 2つの Tensor を相互的にブロードキャストするサイズを求める
    // ※Tensorは四則演算の際などに自動的にブロードキャストが行われないため明示的にこの関数を利用する
    public static int[] broadcastShape(Tensor t0, Tensor t1) {
        return broadcastShape(t0.shape, t1.shape);
    }

    public static int[] broadcastShape(Tensor t0, int[] shape1) {
        return broadcastShape(t0.shape, shape1);
    }

    public static int[] broadcastShape(int[] shape0, Tensor t1) {
        return broadcastShape(shape0, t1.shape);
    }

    public static int[] broadcastShape(int[] shape0, int[] shape1) {
        // 最大の次元数を取得
        int length = Math.max(shape0.length, shape1.length);

        // 結果の形状を格納する配列
        int[] shape = new int[length];

        // shapeA と shapeB を後ろ合わせにするためのオフセットを計算
        int offset0 = length - shape0.length;
        int offset1 = length - shape1.length;

        // 次元ごとにブロードキャストの判定
        for (int i = length - 1; i >= 0; i--) {
            int s0 = (i - offset0 >= 0) ? shape0[i - offset0] : 1;
            int s1 = (i - offset1 >= 0) ? shape1[i - offset1] : 1;

            if (s0 == s1 || s0 == 1 || s1 == 1) {
                // ブロードキャスト可能な次元を選択
                shape[i] = Math.max(s0, s1);
            } else {
                // ブロードキャスト不可能な場合
                System.err.println("Shapes " + Arrays.toString(shape0) +
                        " and " + Arrays.toString(shape1) +
                        " are not broadcastable.");
                return null;
            }
        }

        return shape;
    }

    // ChatGPTで修正
    public static Tensor sumTo(Tensor t, int[] shape) {
        // 縮小後の配列を作成
        double[] values = new double[getLength(shape)];
        int xShape0 = 1;
        int xShape1 = 1;

        // 元の配列のサイズ
        switch (t.rank) {
            case 0:
                break;
            case 1:
                xShape1 = t.getShape()[0];
                break;
            case 2:
                xShape0 = t.getShape()[0];
                xShape1 = t.getShape()[1];
                break;
            default:
                System.err.println(ERROR_RANK);
                throw new RuntimeException(ERROR_RANK);
        }
        // ターゲットの形状
        int shape0 = 1;
        int shape1 = 1;
        switch (shape.length) {
            case 0:
                break;
            case 1:
                shape1 = shape[0];
                break;
            case 2:
                shape0 = shape[0];
                shape1 = shape[1];
                break;
            default:
                System.err.println(ERROR_RANK);
                throw new RuntimeException(ERROR_RANK);
        }

        // 累積和の計算
        for (int i = 0; i < xShape0; i++) {
            for (int j = 0; j < xShape1; j++) {
                // 対応するターゲットのインデックス
                int i_ = i % shape0;
                int j_ = j % shape1;

                // 結果配列に加算
                values[i_ * shape1 + j_] += t.getValues()[i * xShape1 + j];
            }
        }
        return new Tensor(values, shape);
    }

    public static int getLength(int... shape) {
        int length = 1;
        for (int n : shape) {
            length *= n;
        }
        return length;
    }

    public static int[] getIndices(int[] shape, int index) {
        int[] indices = new int[shape.length];
        for (int i = 0; i < shape.length; i++) {
            switch (i) {
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

    public static int getIndex(int[] shape, int... indices) {
        return switch (shape.length) {
            case 0 -> 0;
            case 1 -> indices[0];
            case 2 -> indices[0] * shape[1] + indices[1];
            default -> throw new RuntimeException(ERROR_RANK);
        };
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

    public static boolean compareShape(Tensor t0, Tensor t1) {
        return Arrays.equals(t0.shape, t1.shape);
    }

    public static Tensor caseDifferentShape(Tensor t0, Tensor t1) {
        if (!compareShape(t0, t1)) {
            System.err.print(Arrays.toString(t0.shape));
            System.err.print(Arrays.toString(t1.shape));
            System.err.println("Tensor Shape Error");
        }
        return null;
    }

    public static Tensor sort(Tensor tensor, boolean ascending) {
        double[][] matrix;
        // 行列をコピーしてソート操作
        switch (tensor.getRank()) {
            case 0:
                return tensor.clone();
            case 1:
                matrix = new double[tensor.getLength()][1];
                for (int i = 0; i < tensor.getShape(0); i++) {
                    matrix[i][1] = tensor.getValue(i);
                }
                break;
            case 2:
                matrix = new double[tensor.getShape(0)][tensor.getShape(1)];
                for (int i = 0; i < tensor.getShape(0); i++) {
                    for (int j = 0; j < tensor.getShape(1); j++) {
                        matrix[i][j] = tensor.getValue(i, j);
                    }
                }
            default:
                return null;
        }
        matrix = sort(matrix, ascending);
        if (tensor.getRank() == 1) {
            double[] values = new double[tensor.getLength()];
            for (int i = 0; i < tensor.getLength(); i++) {
                values[i] = matrix[i][0];
            }
            return new Tensor(values);
        } else {
            return new Tensor(matrix);
        }
    }

    public static double[][] sort(double[][] matrix, boolean ascending) {
        // 行列をコピーしてソート操作
        double[][] sortedMatrix = Arrays.copyOf(matrix, matrix.length);

        // ソート処理
        Arrays.sort(sortedMatrix, (row1, row2) -> {
            if (ascending) {
                return Double.compare(row1[0], row2[0]);
            } else {
                return Double.compare(row2[0], row1[0]);
            }
        });
        return sortedMatrix;
    }

    /**
     * 指定した列を基準に行をソートし、その結果の行番号を返す
     *
     * @param tensor    ソート対象の行列
     * @param column    列番号
     * @param ascending 昇順ソートの場合はtrue、降順ソートの場合はfalse
     * @return ソートされた行番号の配列
     */
    public static int[] getSortedIndices(Tensor tensor, int column, boolean ascending) {
        // 行数を取得
        int rowCount = tensor.length;
        double[][] matrix;
        // 行列をコピーしてソート操作
        switch (tensor.getRank()) {
            case 0:
                return new int[]{0};
            case 1:
                matrix = new double[tensor.getLength()][1];
                for (int i = 0; i < tensor.getShape(0); i++) {
                    matrix[i][1] = tensor.getValue(i);
                }
                break;
            case 2:
                matrix = new double[tensor.getShape(0)][tensor.getShape(1)];
                for (int i = 0; i < tensor.getShape(0); i++) {
                    for (int j = 0; j < tensor.getShape(1); j++) {
                        matrix[i][j] = tensor.getValue(i, j);
                    }
                }
            default:
                return null;
        }
        // 行番号の配列を作成
        Integer[] rowIndices = new Integer[rowCount];
        for (int i = 0; i < rowCount; i++) {
            rowIndices[i] = i;
        }

        // ソート処理
        Arrays.sort(rowIndices, (row1, row2) -> {
            if (ascending) {
                return Double.compare(matrix[row1][column], matrix[row2][column]);
            } else {
                return Double.compare(matrix[row2][column], matrix[row1][column]);
            }
        });

        // Integer配列をint配列に変換して返す
        return Arrays.stream(rowIndices).mapToInt(Integer::intValue).toArray();
    }

    /**
     * 指定した列を基準に行をソートし、その結果の行番号を返す
     *
     * @param matrix    ソート対象の行列
     * @param column    列番号
     * @param ascending 昇順ソートの場合はtrue、降順ソートの場合はfalse
     * @return ソートされた行番号の配列
     */
    public static int[] getSortedIndices(double[][] matrix, int column, boolean ascending) {
        // 行数を取得
        int rowCount = matrix.length;

        // 行番号の配列を作成
        Integer[] rowIndices = new Integer[rowCount];
        for (int i = 0; i < rowCount; i++) {
            rowIndices[i] = i;
        }

        // ソート処理
        Arrays.sort(rowIndices, (row1, row2) -> {
            if (ascending) {
                return Double.compare(matrix[row1][column], matrix[row2][column]);
            } else {
                return Double.compare(matrix[row2][column], matrix[row1][column]);
            }
        });

        // Integer配列をint配列に変換して返す
        return Arrays.stream(rowIndices).mapToInt(Integer::intValue).toArray();
    }

    public static int[] getSortedIndices(double[][] matrix, boolean ascending) {
        return getSortedIndices(matrix, 0, ascending);
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
}
