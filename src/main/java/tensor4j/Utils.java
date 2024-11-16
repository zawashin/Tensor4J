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
        return create(0.0, shape);
    }

    public static Tensor create(double value, int... shape) {
        Tensor t = new Tensor(shape);
        for (int i = 0; i < t.length; i++) {
            t.values[i] = value;
        }
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

    public static Tensor to2ndOrder(Tensor t) {
        if (t.rank == 1) {
            int[] shape = new int[]{t.getShape(1), t.getShape(0), t.getShape(2), t.getShape(3)};
            double[][] values = new double[][]{t.getValues()};
            return new Tensor(values);
        } else {
            throw new RuntimeException(Utils.ERROR_RANK);
        }
    }

    public static Tensor transpose(Tensor t, int... axes) {
        int[] shape = new int[Tensor.RANK_MAX];
        Arrays.fill(shape, 1);
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
                tr = Utils.create(t.shape[1], t.shape[0], 1, 1);
                for (int i = 0; i < t.shape[0]; i++) {
                    for (int j = 0; j < t.shape[1]; j++) {
                        tr.setValue(j, i, t.getValue(i, j));
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
        return broadcastTo(t, shape, 0);
    }

    public static Tensor broadcastTo(Tensor t, int[] shape, int axis) {
        try {
            int[] shape_ = new int[Tensor.RANK_MAX];
            Arrays.fill(shape_, 1);
            System.arraycopy(shape, 0, shape_, 0, shape.length);
            int[] xShape = t.getShape();
            Tensor tb = new Tensor(shape_);
            // ブロードキャスト形が整数倍になっているかのチェック
            /*
            for (int i = 0; i < Tensor.RANK_MAX; i++) {
                if(t.shape[i] != (tb.shape[i]/t.shape[i]) * t.shape[i]) {
                    throw new RuntimeException(Utils.ERROR_SHAPE);
                }
            }

             */
            int n = 0;
            switch (t.rank) {
                case 0:
                    for (int i = 0; i < tb.getLength(); i++) {
                        tb.getValues()[i] = t.getValue();
                    }
                    break;
                case 1:
                    switch (tb.rank) {
                        case 0:
                        case 1:
                            throw new RuntimeException();
                        case 2:
                            if (t.shape[0] == tb.shape[1]) {
                                for (int i = 0; i < tb.shape[0]; i++) {
                                    //int i_ = i % t.shape[i];
                                    for (int j = 0; j < t.shape[0]; j++) {
                                        tb.setValue(i, j, t.getValue(j));
                                    }
                                }
                                break;
                            } else if (t.shape[0] == tb.shape[0] && t.shape[1] != tb.shape[1]) {
                                for (int i = 0; i < t.shape[0]; i++) {
                                    for (int j = 0; j < t.shape[0]; j++) {
                                        int j_ = j % t.shape[j];
                                        tb.setValue(i, j, t.getValue(j_));
                                    }
                                }
                                break;
                            } else if (t.shape[0] == tb.shape[0]) {
                                if (axis == 0) {
                                    for (int i = 0; i < tb.shape[0]; i++) {
                                        for (int j = 0; j < t.shape[0]; j++) {
                                            tb.setValue(i, j, t.getValue(j));
                                        }
                                    }
                                } else {
                                    for (int i = 0; i < t.shape[0]; i++) {
                                        for (int j = 0; j < t.shape[0]; j++) {
                                            int j_ = j % t.shape[j];
                                            tb.setValue(i, j, t.getValue(j_));
                                        }
                                    }
                                }
                                break;
                            } else {
                                System.err.println(Arrays.toString(t.shape));
                                System.err.println(Arrays.toString(tb.shape));
                                throw new RuntimeException(Utils.NOT_IMPLEMENTED);
                            }
                    }
                    break;
                case 2:
                    switch (tb.rank) {
                        case 0:
                        case 1:
                            throw new RuntimeException(Utils.NOT_IMPLEMENTED);
                        case 2:
                            for (int i = 0; i < shape[0]; i++) {
                                for (int j = 0; j < shape[1]; j++) {
                                    int i_ = (xShape[0] == 1 || xShape[0] == shape[0]) ? Math.min(i, xShape[0] - 1) : i % xShape[0];
                                    int j_ = (xShape[1] == 1 || xShape[1] == shape[1]) ? Math.min(j, xShape[1] - 1) : j % xShape[1];
                                    tb.getValues()[n++] = t.getValue(i_, j_);
                                }
                            }
                            break;
                        case 3:
                        case 4:
                            throw new RuntimeException(Utils.NOT_IMPLEMENTED);
                    }
                    break;
                case 3:
                case 4:
                    throw new RuntimeException(Utils.NOT_IMPLEMENTED);
                default:
                    break;
            }
            return tb;
        } catch (RuntimeException e) {
            System.err.println(e.getMessage());
            System.exit(1);
        }
        return null;
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
                        int rowIndex = (shape[0] == 1 || shape[0] == xShape[0]) ? Math.min(i, shape[0] - 1) : i % shape[0];
                        int colIndex = (shape[1] == 1 || shape[1] == xShape[1]) ? Math.min(j, shape[1] - 1) : j % shape[1];
                        values[rowIndex * shape[1] + colIndex] += t.getValue(i, j);
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

    // 2つの Tensor を相互的にブロードキャストするサイズを求める
    // ※Tensorは四則演算の際などに自動的にブロードキャストが行われないため明示的にこの関数を利用する
    public static int[] broadcastShape(Tensor t0, Tensor t1) {
        int[] shape = new int[Tensor.RANK_MAX];
        for (int i = 0; i < Tensor.RANK_MAX; i++) {
            shape[i] = Math.max(t0.shape[i], t1.shape[i]);
        }
        return shape;
    }

    public static int[] broadcastShape(Tensor... ts) {
        int[] shape = new int[Tensor.RANK_MAX];
        for (int i = 0; i < Tensor.RANK_MAX; i++) {
            shape[i] = Math.max(ts[0].shape[i], ts[1].shape[i]);
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

    public static int[] validateShape(int... shape) {
        int[] shape_ = new int[Tensor.RANK_MAX];
        Arrays.fill(shape_, 1);
        System.arraycopy(shape, 0, shape_, 0, shape.length);
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < Tensor.RANK_MAX; i++) {
            if (shape_[i] != 1) {
                list.add(shape_[i]);
            }
        }
        if (list.size() == Tensor.RANK_MAX) {
            return shape;
        }
        while (list.size() < Tensor.RANK_MAX) {
            list.add(1);
        }
        return list.stream().mapToInt(Integer::intValue).toArray();
    }

    public static int calcRank(int... shape) {
        int[] shape_;
        if (shape.length != Tensor.RANK_MAX) {
            shape_ = new int[Tensor.RANK_MAX];
            Arrays.fill(shape_, 1);
            System.arraycopy(shape, 0, shape_, 0, shape.length);
        } else {
            shape_ = shape;
        }
        int numOf1 = 0;
        for (int i = 0; i < Tensor.RANK_MAX; i++) {
            if (shape_[i] == 1) {
                numOf1++;
            }
        }
        return Tensor.RANK_MAX - numOf1;
    }
}
