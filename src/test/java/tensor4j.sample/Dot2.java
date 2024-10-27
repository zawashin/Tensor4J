package tensor4j.sample;

import tensor4j.Tensor;

import java.util.Arrays;

import static tensor4j.Utils.create;

/**
 * @author Shin-Ichiro Serizawa <zawashin@outlook.com>
 */
public class Dot2 {
    public static void main(String[] args) {
        Tensor t1;
        Tensor t2;
        {
            double[] a1 = {1.0, 2.0, 3.0};
            double[][] a2 = {
                    {4.0, 5.0, 6.0},
                    {7.0, 8.0, 9.0},
                    {10.0, 11.0, 12.0}
            };
            t1 = new Tensor(a1);
            t2 = new Tensor(a2);
            System.out.println(t1);
            System.out.println(t2);
            System.out.println(Arrays.toString(t1.dot(t2).getShape()));
            System.out.println(t1.dot(t2));
        }
        {
            double[][] a1 = {
                    {1.0, 2.0, 3.0},
                    {4.0, 5.0, 6.0},
                    {7.0, 8.0, 9.0}
            };
            double[] a2 = {10.0, 11.0, 12.0};
            t1 = new Tensor(a1);
            t2 = new Tensor(a2);
            System.out.println(t1);
            System.out.println(t2);
            System.out.println(Arrays.toString(t1.dot(t2).getShape()));
            System.out.println(t1.dot(t2));
        }
        {
            double[][] a1 = {
                    {1.0, 2.0, 3.0},
                    {4.0, 5.0, 6.0},
                    {7.0, 8.0, 9.0}
            };
            double[][] a2 = {
                    {10.0, 11.0},
                    {12.0, 13.0},
                    {14.0, 15.0}
            };
            t1 = new Tensor(a1);
            t2 = new Tensor(a2);
            System.out.println(t1);
            System.out.println(t2);
            System.out.println(Arrays.toString(t1.dot(t2).getShape()));
            System.out.println(t1.dot(t2));
        }
    }
}
