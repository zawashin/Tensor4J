package tensor4j.FileIO;

/**
 * @author Shin-Ichiro Serizawa <zawashin@outlook.com>
 */
public class ArrayMinMaxScaler {

    public double minRange;
    public double maxRange;
    double dRange;
    double minValue;
    double maxValue;
    double dValue;

    public ArrayMinMaxScaler() {
        minRange = 0.0;
        maxRange = 1.0;
        dRange = maxRange - minRange;
    }

    public ArrayMinMaxScaler(double minRange, double maxRange) {
        if (minRange == maxRange) {
            System.out.println("min equals max");
            try {
                throw new Exception();
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(1);
            }
        }
        this.minRange = minRange;
        this.maxRange = maxRange;
        dRange = maxRange - minRange;
    }

    public void transform(double[] values) {
        minValue = Double.MAX_VALUE;
        maxValue = -Double.MAX_VALUE;
        for (int i = 0; i < values.length; i++) {
            if (values[i] < minValue) {
                minValue = values[i];
            } else if (values[i] > maxValue) {
                maxValue = values[i];
            }
        }
        dValue = maxValue - minValue;
        for (int i = 0; i < values.length; i++) {
            values[i] = dRange * (values[i] - minValue) / dValue + minRange;
        }
    }

    public void transform(double[][] values) {
        minValue = Double.MAX_VALUE;
        maxValue = -Double.MAX_VALUE;
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[i].length; j++) {
                if (values[i][j] < minValue) {
                    minValue = values[i][j];
                } else if (values[i][j] > maxValue) {
                    maxValue = values[i][j];
                }
            }
        }
        dValue = maxValue - minValue;
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[i].length; j++) {
                values[i][j] = dRange * (values[i][j] - minValue) / dValue + minRange;
            }
        }
    }

    public void logTransform(double[] values) {
        minValue = Double.MAX_VALUE;
        maxValue = -Double.MAX_VALUE;
        for (int i = 0; i < values.length; i++) {
            if (values[i] < minValue) {
                minValue = values[i];
            } else if (values[i] > maxValue) {
                maxValue = values[i];
            }
        }
        dValue = maxValue - minValue;
        for (int i = 0; i < values.length; i++) {
            //values[i] = dRange * (values[i] - minValue) / dValue + minRange;
        }
    }

    public void logTransform(double[][] values) {
        minValue = Double.MAX_VALUE;
        maxValue = -Double.MAX_VALUE;
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[i].length; j++) {
                if (values[i][j] < minValue) {
                    minValue = values[i][j];
                } else if (values[i][j] > maxValue) {
                    maxValue = values[i][j];
                }
            }
        }
        dValue = maxValue - minValue;
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[i].length; j++) {
                //values[i][j] = dRange * (values[i][j] - minValue) / dValue + minRange;
            }
        }
    }

    public void revert(double[] values) {
        for (int i = 0; i < values.length; i++) {
            //values[i] = (values[i] - minRange) / dRange * dValue + minValue;
        }
    }

    public void revert(double[][] values) {
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[i].length; j++) {
                //values[i][j] = (values[i][j] - minRange) / dRange * dValue + minValue;
            }
        }
    }

    public void logRevert(double[] values) {
        for (int i = 0; i < values.length; i++) {
            values[i] = (values[i] - minRange) / dRange * dValue + minValue;
        }
    }

    public void logRevert(double[][] values) {
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[i].length; j++) {
                values[i][j] = (values[i][j] - minRange) / dRange * dValue + minValue;
            }
        }
    }
}
