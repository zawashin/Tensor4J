package tensor4j.sample;

import tensor4j.Tensor;

import java.util.Arrays;

/**
 * @author Shin-Ichiro Serizawa <zawashin@outlook.com>
 */
public class ToString {
    public static void main(String[] args) {
        Tensor tensor;
        int n;
        System.out.println("1D");
        double[] vector = new double[5];
        tensor = new Tensor(vector);
        n = 0;
        for (int i = 0; i < tensor.getLength(); i++) {
            tensor.setValue(i, n++ + 1);
        }
        System.out.println(Arrays.toString(tensor.getShape()));
        System.out.println(tensor);

        double[][] matrix = new double[3][5];
        System.out.println("2D");
        tensor = new Tensor(matrix);
        n = 0;
        for (int i = 0; i < tensor.getLength(); i++) {
            tensor.getValues()[n] = n++;
        }
        for (int i = 0; i < tensor.getShape(0); i++) {
            for (int j = 0; j < tensor.getShape(1); j++) {
                tensor.setValue(i, j, (i + 1) * 10 + (j + 1));
            }
        }
        System.out.println(Arrays.toString(tensor.getShape()));
        System.out.println(tensor);
        Tensor tr = tensor.transpose();
        System.out.println(Arrays.toString(tr.getShape()));
        System.out.println(tr);

        double[][][] array = new double[3][5][2];
        tensor = new Tensor(array);
        for (int i = 0; i < tensor.getShape(0); i++) {
            for (int j = 0; j < tensor.getShape(1); j++) {
                for (int k = 0; k < tensor.getShape(2); k++) {
                    tensor.setValue(i, j, k, (i + 1) * 100 + (j + 1) * 10 + (k + 1));
                }
            }
        }
        System.out.println(Arrays.toString(tensor.getShape()));
        System.out.println(tensor);
         /*
        tr = tensor.transpose();
        System.out.println(Arrays.toString(tr.shape));
        System.out.println(tr);
         */

        double[][][][] array2 = new double[2][5][3][2];
        tensor = new Tensor(array2);
        for (int i = 0; i < tensor.getShape(0); i++) {
            for (int j = 0; j < tensor.getShape(1); j++) {
                for (int k = 0; k < tensor.getShape(2); k++) {
                    for (int l = 0; l < tensor.getShape(3); l++) {
                        tensor.setValue(i, j, k, l, (i + 1) * 1000 + (j + 1) * 100 + (k + 1) * 10 + l + 1);
                    }
                }
            }
        }
        System.out.println(Arrays.toString(tensor.getShape()));
        System.out.println(tensor);
    }

}
