package tensor4j.sample;

import tensor4j.Tensor;

/**
 * @author Shin-Ichiro Serizawa <zawashin@outlook.com>
 */
public class ArithmeticFunction {

    public static void main(String[] args) {
        double[][] m0 = new double[3][2];
        for(int i = 0; i < m0.length; i++) {
            for(int j = 0; j< m0[0].length; j++) {
                m0[i][j] = (i + 1) * (j + 1);
            }
        }
        Tensor t0 = new Tensor(m0);

        System.out.println(t0);
        System.out.println(t0.log());
        System.out.println(t0.pow(2));

    }
}
