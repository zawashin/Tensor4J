package tensor4j;

import java.util.Arrays;

/**
 * @author Shin-Ichiro Serizawa <zawashin@outlook.com>
 */

public class Sort {
    public static void main(String[] args) {
        // サンプル行列
        double[][] matrix = {
                {3.5, 2.1, 4.0},
                {1.2, 3.3, 5.1},
                {1.4, 3.3, 5.1},
                {2.8, 1.5, 3.9}
        };

        // 昇順にソート
        System.out.println("昇順ソート:");
        double[][] ascendingSorted = sort(matrix, true);
        printMatrix(ascendingSorted);

        // 降順にソート
        System.out.println("\n降順ソート:");
        double[][] descendingSorted = sort(matrix, false);
        printMatrix(descendingSorted);
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

    public static void printMatrix(double[][] matrix) {
        for (double[] row : matrix) {
            System.out.println(Arrays.toString(row));
        }
    }
}
