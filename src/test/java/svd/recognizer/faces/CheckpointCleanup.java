package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.DirectoryNotEmptyException;

/**
 * Удаление каталога контрольной точки в тестах. В Windows только что записанный файл ненадолго открывает внешний
 * процесс (проверка файлов); если удалить каталог в это окно, Files.delete(каталог) даёт DirectoryNotEmptyException,
 * хотя файлы уже удалены, а через миллисекунды каталог удаляется. Поэтому — повтор с паузой; прочие ошибки не глушатся.
 *
 * @author ssv
 */
final class CheckpointCleanup {

    private static final int ATTEMPTS = 100;
    private static final long PAUSE_MS = 10;

    private CheckpointCleanup() {
    }

    static void delete(FeiCheckpoint cp) throws IOException, InterruptedException {
        for (int i = 1;; i++) {
            try {
                cp.delete();
                return;
            } catch (DirectoryNotEmptyException e) {
                if (i == ATTEMPTS) throw e;
                Thread.sleep(PAUSE_MS);
            }
        }
    }
}
