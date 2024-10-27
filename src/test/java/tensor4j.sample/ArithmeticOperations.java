package tensor4j.sample;

import tensor4j.Tensor;

/**
 * @author Shin-Ichiro Serizawa <zawashin@outlook.com>
 */
public class ArithmeticOperations {
    public static void main(String[] args) {
        double[][] m0 = new double[3][2];
        double[][] m1 = new double[3][2];
        for(int i = 0; i < m0.length; i++) {
            for(int j = 0; j< m0[0].length; j++) {
                m0[i][j] = i + 1;
                m1[i][j] = j + 1;
            }
        }
        Tensor t0 = new Tensor(m0);
        Tensor t1 = new Tensor(m1);

        System.out.println(t0);
        System.out.println(t1);
        System.out.println(t0.plus(t1));
        System.out.println(t0.plus(1));
        System.out.println(t0.minus(t1));
        System.out.println(t0.minus(1));
        System.out.println(t0.times(t1));
        System.out.println(t0.div(t1));

    }
}
