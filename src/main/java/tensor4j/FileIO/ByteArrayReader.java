package tensor4j.FileIO;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * @author shin
 */
public class ByteArrayReader {

    String filePath;

    /**
     * @param filePath
     */
    public ByteArrayReader(String filePath) {
        this.filePath = filePath;
    }

    /**
     * @return
     */
    public byte[] read() {
        FileInputStream fileInStm = null;
        try {
            fileInStm = new FileInputStream(filePath);
        } catch (FileNotFoundException e1) {
            System.out.println("ファイルが見つからなかった。");
        }
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte[] buffer = new byte[1024];
        while (true) {
            int len = 0;
            try {
                len = fileInStm.read(buffer);
            } catch (IOException e) {
                e.printStackTrace();
            }
            if (len < 0) {
                break;
            }
            baos.write(buffer, 0, len);
        }
        return baos.toByteArray();
    }
}
