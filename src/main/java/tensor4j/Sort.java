package tensor4j;

import java.util.Arrays;

/**
 * @author Shin-Ichiro Serizawa <zawashin@outlook.com>
 */

public class Sort {

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

    public static void printMatrix(double[][] matrix) {
        for (double[] row : matrix) {
            System.out.println(Arrays.toString(row));
        }
    }
}
