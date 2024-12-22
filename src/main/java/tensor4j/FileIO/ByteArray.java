package tensor4j.FileIO;

import java.io.*;

/**
 * @author shin
 */
public final class ByteArray {

    /**
     * @param object
     * @return
     * @throws IOException
     */
    public static byte[] fromObject(Object object) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        ObjectOutput out = new ObjectOutputStream(bos);
        out.writeObject(object);
        byte[] bytes = bos.toByteArray();
        out.close();
        bos.close();
        return bytes;
    }

    /**
     * @param bytes
     * @return
     * @throws ClassNotFoundException
     * @throws IOException
     */
    public static Object toObject(byte[] bytes) throws ClassNotFoundException, IOException {
        return new ObjectInputStream(new ByteArrayInputStream(bytes)).readObject();
    }

    /*
    public static void write(Object object, String filePath) {
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
    public static byte[] read(String filePath) {
        FileInputStream fileInStm = null;
        try {
            fileInStm = new FileInputStream("test.dat");
        } catch (FileNotFoundException e1) {
            System.out.println("ファイルが見つからなかった。");
        }
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte [] buffer = new byte[1024];
        while(true) {
            int len = 0;
            try {
                len = fileInStm.read(buffer);
            } catch (IOException e) {
                e.printStackTrace();
            }
            if(len < 0) {
                break;
            }
            baos.write(buffer, 0, len);
        }
        return baos.toByteArray();
    }
    */
}
