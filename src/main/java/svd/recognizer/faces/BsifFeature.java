package svd.recognizer.faces;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.DataFormatException;
import java.util.zip.Inflater;

/**
 * Гистограммы BSIF (Kannala, Rahtu, «BSIF: Binarized statistical image features», ICPR 2012; пачка 2, задание Жорика №2 от
 * 28.10.2026). Фильтры – из архива авторов (bsif_code_and_data.zip, texturefilters/ICAtextureFilters_{n}x{n}_{b}bit.mat, обучены
 * ICA на естественных изображениях; каталог – ключ faces.bsif.dir / FACES_BSIF_DIR, SOURCE.txt там же). Код – как bsif.m: отклик
 * i-го бита (i = 1…b) – корреляция (filter2) кадра с фильтром номер b − i + 1, бит 1 при отклике > 0, метки 0…2^b − 1.
 * Отступление от авторов (решение Жорика №3 от 28.10): коды – только у пикселей с окном целиком в кадре (край ⌊n/2⌋), как у
 * LbpFeature и PoemFeature; у авторов край продолжается циклически. Сетка блоков, нормировка и порядок – как у LpqFeature.
 *
 * @author ssv
 */
final class BsifFeature {

    /** Фильтры: [номер][строка][столбец] (порядок файла, номер 0 – первый). */
    final double[][][] filters;
    final int n;
    final int bits;

    BsifFeature(double[][][] filters) {
        this.filters = filters;
        this.bits = filters.length;
        this.n = filters[0].length;
        if (bits > 16) throw new IllegalArgumentException("BSIF: бит " + bits);
    }

    int length(int grid) {
        return grid * grid * (1 << bits);
    }

    static Path file(Path dir, int n, int bits) {
        return dir.resolve("bsif_code_and_data").resolve("texturefilters").resolve("ICAtextureFilters_" + n + "x" + n + "_" + bits + "bit.mat");
    }

    static BsifFeature load(Path dir, int n, int bits) throws IOException {
        double[][][] f = readFilters(Files.readAllBytes(file(dir, n, bits)), "ICAtextureFilters");
        if (f.length != bits || f[0].length != n || f[0][0].length != n) {
            throw new IOException("BSIF: в " + file(dir, n, bits) + " фильтры " + f[0].length + "×" + f[0][0].length + "×" + f.length);
        }
        return new BsifFeature(f);
    }

    /** Коды пикселей с окном целиком в кадре: [(h − 2r)·(w − 2r)] по строкам, r = ⌊n/2⌋. */
    int[] codes(double[] img, int w, int h) {
        int r = n / 2;
        int cw = w - 2 * r;
        int ch = h - 2 * r;
        if (cw <= 0 || ch <= 0) throw new IllegalArgumentException("BSIF: кадр " + w + "×" + h + " меньше окна");
        int[] out = new int[cw * ch];
        for (int i = 1; i <= bits; i++) {
            double[][] t = filters[bits - i];
            for (int y = 0; y < ch; y++) {
                for (int x = 0; x < cw; x++) {
                    double s = 0;
                    for (int u = 0; u < n; u++) {
                        int row = (y + u) * w + x;
                        for (int v = 0; v < n; v++) s += t[u][v] * img[row + v];
                    }
                    if (s > 0) out[y * cw + x] |= 1 << (i - 1);
                }
            }
        }
        return out;
    }

    /** Вектор гистограмм (length(grid)) по кадру img. */
    double[] histogram(double[] img, int w, int h, int grid) {
        int r = n / 2;
        return LpqFeature.blocks(codes(img, w, h), w - 2 * r, h - 2 * r, grid, 1 << bits);
    }

    // ---------------------------------------------------------------- MAT-файл (уровень 5)

    static final int MI_INT8 = 1;
    static final int MI_UINT8 = 2;
    static final int MI_INT16 = 3;
    static final int MI_UINT16 = 4;
    static final int MI_INT32 = 5;
    static final int MI_UINT32 = 6;
    static final int MI_SINGLE = 7;
    static final int MI_DOUBLE = 9;
    static final int MI_MATRIX = 14;
    static final int MI_COMPRESSED = 15;

    /**
     * Трёхмерный вещественный массив name из MAT-файла уровня 5 (MATLAB 5.0; элементы miCOMPRESSED распаковываются): [k][строка][столбец],
     * данные MATLAB – по столбцам (элемент (r, c, k) – номер r + rows·c + rows·cols·k).
     */
    static double[][][] readFilters(byte[] file, String name) throws IOException {
        if (file.length < 128) throw new IOException("MAT: короткий файл");
        ByteOrder order = file[126] == 'I' && file[127] == 'M' ? ByteOrder.LITTLE_ENDIAN : ByteOrder.BIG_ENDIAN;
        ByteBuffer b = ByteBuffer.wrap(file, 128, file.length - 128).order(order);
        while (b.remaining() >= 8) {
            int type = b.getInt();
            int size = b.getInt();
            byte[] data = new byte[size];
            b.get(data);
            if (type == MI_COMPRESSED) {
                data = inflate(data);
                ByteBuffer in = ByteBuffer.wrap(data).order(order);
                type = in.getInt();
                size = in.getInt();
                data = new byte[size];
                in.get(data);
            } else if (b.remaining() > 0) {
                b.position(b.position() + pad(size));
            }
            if (type != MI_MATRIX) continue;
            double[][][] a = matrix(ByteBuffer.wrap(data).order(order), name);
            if (a != null) return a;
        }
        throw new IOException("MAT: нет массива " + name);
    }

    static int pad(int size) {
        return (8 - size % 8) % 8;
    }

    static byte[] inflate(byte[] z) throws IOException {
        Inflater inf = new Inflater();
        inf.setInput(z);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buf = new byte[1 << 16];
        try {
            while (!inf.finished()) {
                int k = inf.inflate(buf);
                if (k == 0 && (inf.needsInput() || inf.needsDictionary())) break;
                out.write(buf, 0, k);
            }
        } catch (DataFormatException e) {
            throw new IOException("MAT: сжатие " + e, e);
        } finally {
            inf.end();
        }
        return out.toByteArray();
    }

    /** Подэлемент: [тип, данные]; малый формат (данные ≤ 4 байт в теге) – тоже. */
    static Object[] element(ByteBuffer b) {
        int tag = b.getInt();
        if ((tag >>> 16) != 0) {
            int size = tag >>> 16;
            byte[] d = new byte[size];
            b.get(d);
            b.position(b.position() + 4 - size);
            return new Object[] {tag & 0xFFFF, d};
        }
        int size = b.getInt();
        byte[] d = new byte[size];
        b.get(d);
        b.position(Math.min(b.limit(), b.position() + pad(size)));
        return new Object[] {tag, d};
    }

    static double[][][] matrix(ByteBuffer b, String name) throws IOException {
        Object[] flags = element(b);
        byte[] fl = (byte[]) flags[1];
        int cls = ByteBuffer.wrap(fl).order(b.order()).getInt() & 0xFF;
        if (cls != 6 && cls != 7) return null; // mxDOUBLE_CLASS = 6, mxSINGLE_CLASS = 7
        if ((ByteBuffer.wrap(fl).order(b.order()).getInt() & 0x800) != 0) throw new IOException("MAT: комплексный массив");
        Object[] dimsE = element(b);
        ByteBuffer dd = ByteBuffer.wrap((byte[]) dimsE[1]).order(b.order());
        int[] dims = new int[((byte[]) dimsE[1]).length / 4];
        for (int i = 0; i < dims.length; i++) dims[i] = dd.getInt();
        String nm = new String((byte[]) element(b)[1], java.nio.charset.StandardCharsets.US_ASCII);
        if (!nm.equals(name)) return null;
        Object[] re = element(b);
        double[] v = values((int) re[0], (byte[]) re[1], b.order());
        int rows = dims[0];
        int cols = dims.length > 1 ? dims[1] : 1;
        int pages = dims.length > 2 ? dims[2] : 1;
        if (v.length != rows * cols * pages) throw new IOException("MAT: размер данных " + v.length + " при " + rows + "×" + cols + "×" + pages);
        double[][][] a = new double[pages][rows][cols];
        for (int k = 0; k < pages; k++) for (int c = 0; c < cols; c++) for (int r = 0; r < rows; r++) a[k][r][c] = v[r + rows * c + rows * cols * k];
        return a;
    }

    static double[] values(int type, byte[] d, ByteOrder order) throws IOException {
        ByteBuffer b = ByteBuffer.wrap(d).order(order);
        return switch (type) {
            case MI_DOUBLE -> {
                double[] v = new double[d.length / 8];
                for (int i = 0; i < v.length; i++) v[i] = b.getDouble();
                yield v;
            }
            case MI_SINGLE -> {
                double[] v = new double[d.length / 4];
                for (int i = 0; i < v.length; i++) v[i] = b.getFloat();
                yield v;
            }
            case MI_INT8 -> {
                double[] v = new double[d.length];
                for (int i = 0; i < v.length; i++) v[i] = d[i];
                yield v;
            }
            case MI_UINT8 -> {
                double[] v = new double[d.length];
                for (int i = 0; i < v.length; i++) v[i] = d[i] & 0xFF;
                yield v;
            }
            case MI_INT16 -> {
                double[] v = new double[d.length / 2];
                for (int i = 0; i < v.length; i++) v[i] = b.getShort();
                yield v;
            }
            case MI_UINT16 -> {
                double[] v = new double[d.length / 2];
                for (int i = 0; i < v.length; i++) v[i] = b.getShort() & 0xFFFF;
                yield v;
            }
            case MI_INT32 -> {
                double[] v = new double[d.length / 4];
                for (int i = 0; i < v.length; i++) v[i] = b.getInt();
                yield v;
            }
            case MI_UINT32 -> {
                double[] v = new double[d.length / 4];
                for (int i = 0; i < v.length; i++) v[i] = b.getInt() & 0xFFFFFFFFL;
                yield v;
            }
            default -> throw new IOException("MAT: тип данных " + type);
        };
    }
}
