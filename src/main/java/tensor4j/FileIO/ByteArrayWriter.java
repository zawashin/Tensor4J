package tensor4j.FileIO;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * @author shin
 */
public class ByteArrayWriter {

    String filePath;

    /**
     * @param filePath
     */
    public ByteArrayWriter(String filePath) {
        this.filePath = filePath;
    }

    /**
     * @param bytes
     */
    public void write(byte[] bytes) {
        FileOutputStream fileOutStm = null;
        try {
            fileOutStm = new FileOutputStream(filePath);
        } catch (FileNotFoundException e1) {
            System.out.println("ファイルが見つからなかった。");
        }
        try {
            fileOutStm.write(bytes);
        } catch (IOException e) {
            System.out.println("入出力エラー。");
        }
        try {
            fileOutStm.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * @param object
     */
    public void write(Object object) {
        byte[] bytes = null;
        try {
            bytes = ByteArray.fromObject(object);
        } catch (IOException e) {
            e.printStackTrace();
        }
        FileOutputStream fileOutStm = null;
        try {
            fileOutStm = new FileOutputStream(filePath);
        } catch (FileNotFoundException e1) {
            System.out.println("ファイルが見つからなかった。");
        }
        try {
            fileOutStm.write(bytes);
        } catch (IOException e) {
            System.out.println("入出力エラー。");
        }
        try {
            fileOutStm.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
