package tensor4j.FileIO;

/**
 * @author shin
 */
public class ArrayIndex {

    /**
     * @param array
     * @return
     */
    public static int IndexOfMexValue(int[] array) {
        int iMax = 0;
        int valueMax = array[0];

        for (int i = 1; i < array.length; i++) {
            if (array[i] > valueMax) {
                iMax = i;
                valueMax = array[i];
            }
        }
        return iMax;
    }

    /**
     * @param array
     * @return
     */
    public static int IndexOfMinValue(int[] array) {
        int iMin = 0;
        int valueMin = array[0];

        for (int i = 1; i < array.length; i++) {
            if (array[i] < valueMin) {
                iMin = i;
                valueMin = array[i];
            }
        }
        return iMin;
    }

    /**
     * @param array
     * @return
     */
    public static int IndexOfMexValue(double[] array) {
        int iMax = 0;
        double valueMax = array[0];

        for (int i = 1; i < array.length; i++) {
            if (array[i] > valueMax) {
                iMax = i;
                valueMax = array[i];
            }
        }
        return iMax;
    }

    /**
     * @param array
     * @return
     */
    public static int IndexOfMinValue(double[] array) {
        int iMin = 0;
        double valueMin = array[0];

        for (int i = 1; i < array.length; i++) {
            if (array[i] < valueMin) {
                iMin = i;
                valueMin = array[i];
            }
        }
        return iMin;
    }

}
