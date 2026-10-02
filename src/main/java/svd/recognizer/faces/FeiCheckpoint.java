package svd.recognizer.faces;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Контрольные точки faces-fei (требование Хозяина 02.10.2026, CLAUDE.md): после каждой законченной конфигурации на диск
 * пишутся её оценки (conf_NN.tsv.gz) и накопленное состояние (state.ser: итоги методов для fei_methods.txt, время, строки
 * коммитов для шапок); повторный запуск той же командой продолжает со следующей конфигурации. Запись — через временный
 * файл и переименование (обрыв посреди записи не портит сохранённое). С состоянием хранится отпечаток (код src и pom.xml
 * с незакоммиченными правками, данные a/fei и fei_selection.tsv, ключи, seed, число конфигураций); не совпал —
 * продолжение запрещено. После сборки итоговых отчётов каталог удаляется.
 *
 * @author ssv
 */
final class FeiCheckpoint {

    static final String DIR = "checkpoint";
    static final String STATE = "state.ser";

    /** Накопленное состояние прогона после done законченных конфигураций. */
    static final class State implements Serializable {
        private static final long serialVersionUID = 1L;
        final String fingerprint;
        /** Строки коммитов кода и источника данных на момент первого запуска (для шапок отчётов). */
        final String code;
        final String source;
        int done;
        LinkedHashMap<String, FeiMethods.Result> results = new LinkedHashMap<>();
        String timeText = "";

        State(String fingerprint, String code, String source) {
            this.fingerprint = fingerprint;
            this.code = code;
            this.source = source;
        }
    }

    final Path dir;

    FeiCheckpoint(Path dir) {
        this.dir = dir;
    }

    /**
     * Состояние для продолжения: нет контрольной точки — null; отпечаток не совпал или части не все — ошибка.
     */
    State load(String fingerprint) throws IOException {
        Path f = dir.resolve(STATE);
        if (!Files.exists(f)) return null;
        State s;
        try (ObjectInputStream in = new ObjectInputStream(Files.newInputStream(f))) {
            s = (State) in.readObject();
        } catch (ClassNotFoundException | ClassCastException | java.io.InvalidClassException e) {
            throw new IllegalStateException("Контрольная точка " + f + " не читается (" + e + "); удалите каталог " + dir
                    + ", чтобы начать прогон заново");
        }
        if (!s.fingerprint.equals(fingerprint)) {
            throw new IllegalStateException("Контрольная точка " + dir + " от другого прогона — продолжать нельзя.\nСохранённый отпечаток:\n"
                    + s.fingerprint + "\nТекущий:\n" + fingerprint + "\nВерните код, данные и ключи как были или удалите каталог " + dir
                    + ", чтобы начать заново.");
        }
        for (int c = 0; c < s.done; c++) {
            if (!Files.exists(part(c))) throw new IllegalStateException("Контрольная точка неполна: нет " + part(c) + "; удалите " + dir);
        }
        return s;
    }

    /** Сохранить оценки законченной конфигурации c (до сохранения состояния). */
    void savePart(int c, String text) throws IOException {
        Files.createDirectories(dir);
        Path tmp = dir.resolve(part(c).getFileName() + ".tmp");
        try (Writer w = new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(tmp), 1 << 16), StandardCharsets.UTF_8)) {
            w.write(text);
        }
        move(tmp, part(c));
    }

    /** Сохранить состояние (после части). */
    void save(State s) throws IOException {
        Files.createDirectories(dir);
        Path tmp = dir.resolve(STATE + ".tmp");
        try (OutputStream o = Files.newOutputStream(tmp); ObjectOutputStream out = new ObjectOutputStream(o)) {
            out.writeObject(s);
        }
        move(tmp, dir.resolve(STATE));
    }

    Path part(int c) {
        return dir.resolve(String.format(Locale.ROOT, "conf_%02d.tsv.gz", c));
    }

    /** Итоговый файл оценок: шапка и части 0…n−1 подряд (содержимое — как при записи без контрольных точек). */
    void assemble(Path file, String header, int n) throws IOException {
        Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
        try (Writer w = new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(tmp), 1 << 16), StandardCharsets.UTF_8)) {
            w.write(header);
            for (int c = 0; c < n; c++) {
                try (InputStream in = new GZIPInputStream(Files.newInputStream(part(c)), 1 << 16)) {
                    w.write(new String(in.readAllBytes(), StandardCharsets.UTF_8));
                }
            }
        }
        move(tmp, file);
    }

    /** Удалить каталог контрольной точки (после записи всех отчётов). */
    void delete() throws IOException {
        if (!Files.exists(dir)) return;
        try (Stream<Path> s = Files.list(dir)) {
            for (Path p : s.toList()) Files.delete(p);
        }
        Files.delete(dir);
    }

    private static void move(Path from, Path to) throws IOException {
        try {
            Files.move(from, to, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (java.nio.file.AtomicMoveNotSupportedException e) {
            Files.move(from, to, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    // ---------------------------------------------------------------- отпечаток

    /**
     * Отпечаток прогона: код (деревья src и pom.xml в HEAD и хеш незакоммиченных правок в них — коммиты документов не
     * мешают продолжению), данные (деревья a/fei и fei_selection.tsv в HEAD репозитория данных и хеш их незакоммиченных
     * изменений), прочее (ключи, seed, число конфигураций).
     */
    static String fingerprint(Path exportDir, String other) {
        List<String> l = new ArrayList<>();
        l.add("код: src " + git(null, "rev-parse", "HEAD:src") + ", pom.xml " + git(null, "rev-parse", "HEAD:pom.xml") + ", правки "
                + sha(git(null, "diff", "HEAD", "--", "src", "pom.xml")));
        l.add("данные: a/fei " + git(exportDir, "rev-parse", "HEAD:a/fei") + ", fei_selection.tsv "
                + git(exportDir, "rev-parse", "HEAD:" + FarExport.FEI_SELECTION) + ", правки "
                + sha(git(exportDir, "status", "--porcelain", "--", "a/fei", FarExport.FEI_SELECTION)));
        l.add(other);
        return String.join("\n", l);
    }

    static String sha(String s) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(s.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    private static String git(Path dir, String... args) {
        List<String> cmd = new ArrayList<>(List.of("git"));
        if (dir != null) cmd.addAll(List.of("-C", dir.toString()));
        cmd.addAll(List.of(args));
        try {
            Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
            String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
            if (p.waitFor() != 0) throw new IllegalStateException("git " + String.join(" ", args) + ": " + out);
            return out;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(e);
        }
    }
}
