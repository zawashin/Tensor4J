package tensor4j.sample;

import tensor4j.Tensor;

import java.util.Arrays;

/**
 * @author Shin-Ichiro Serizawa <zawashin@outlook.com>
 */
public class Transpose {
    public static void main(String[] args) {
        Tensor tensor;
        int n;
        System.out.println("1D");
        double[] vector = new double[5];
        tensor = new Tensor(vector);
        n = 0;
        for(int i = 0; i < tensor.getLength(); i++) {
            tensor.getValues()[n] = n++;
        }
        System.out.println(Arrays.toString(tensor.getShape()));
        System.out.println(tensor);

        double[][] matrix = new double[3][5];
        System.out.println("2D");
        tensor = new Tensor(matrix);
        n = 0;
        for(int i = 0; i < tensor.getLength(); i++) {
            tensor.getValues()[n] = n++;
        }
        System.out.println(Arrays.toString(tensor.getShape()));
        System.out.println(tensor);
        System.out.println(tensor.transpose());

        /*
         */
        double[][][] array = new double[2][5][3];
        tensor = new Tensor(array);
        n = 0;
        for(int i = 0; i < tensor.getLength(); i++) {
            tensor.getValues()[n] = n++;
        }
        System.out.println(Arrays.toString(tensor.getShape()));
        /*
        System.out.println(array[0][0].length);
        System.out.println(array[0].length);
        System.out.println(array.length);

         */
        System.out.println(tensor.getImax());
        System.out.println(tensor.getJmax());
        System.out.println(tensor.getKmax());
        System.out.println(tensor.getLmax());
        System.out.println(tensor);
    }

}
