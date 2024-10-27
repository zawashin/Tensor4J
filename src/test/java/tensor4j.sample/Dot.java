package tensor4j.sample;

import tensor4j.Tensor;
import tensor4j.Utils;

import java.util.Arrays;

import static tensor4j.Utils.create;

/**
 * @author Shin-Ichiro Serizawa <zawashin@outlook.com>
 */
public class Dot {
    public static void main(String[] args) {
        Tensor t13 = new Tensor(new double[3]);
        Tensor t15 = new Tensor(new double[5]);
        System.out.println(Arrays.toString(t13.getShape()));
        System.out.println(Arrays.toString(t15.getShape()));
        Arrays.fill(t13.getValues(), 1);
        Arrays.fill(t15.getValues(), 1);
        Tensor t2;
        int n;
        double[][] matrix = new double[3][5];
        t2 = Utils.create(3, 5);
        n = 1;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                t2.setValue(i, j, n++);
                //matrix[i][j] = n++;//(i + 1) * 10 + (j + 1);
            }
        }
        System.out.println(Arrays.toString(t2.getShape()));
        System.out.println(t2);
        System.out.println(t13.dot(t13));
        System.out.println(t15.dot(t15));
        System.out.println(t13.dot(t2));
        System.out.println(t2.dot(t15));
        Tensor tr = t2.transpose();
        System.out.println(tr);
        System.out.println(t2.dot(tr));
        System.out.println(tr.dot(t2));
    }
}
