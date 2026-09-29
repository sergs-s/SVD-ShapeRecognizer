package svd.recognizer.faces;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.FaceRecognizerSF;
import svd.recognizer.faces.GalleryEvaluation.Base;
import svd.recognizer.faces.GalleryEvaluation.Data;
import svd.recognizer.faces.GalleryEvaluation.Det;
import svd.recognizer.faces.GalleryEvaluation.Feat;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.GalleryEvaluation.Splits;
import svd.recognizer.faces.GalleryEvaluation.Variant;
import svd.recognizer.storage.SettingsStore;

/**
 * Экспорт выровненных кадров для облачных расчётов (шаг 5, данные для облака; одна команда:
 * {@code mvn compile exec:exec@faces-export}, каталог — faces.export.dir / FACES_EXPORT_DIR, репозиторий данных
 * SVD-faces-data). По детекциям faces-far (кэш reports/faces/far/far_detections.cache; пересчёт —
 * {@code -Dfar.args=fresh}) на каждый снимок своей базы, Georgia Tech, MUCT и ORL пишутся PNG без потерь — ровно
 * кадры, из которых GalleryEvaluation строит векторы: a/ — вариант а) 92×112 серый, b/ — б) 92×112 серый,
 * c/ — в) 184×224 серый, sface/ — вход SFace, alignCrop 112×112 BGR; путь — вариант/база/человек/имя.png.
 * Рядом: detections.tsv (все снимки по порядку оценки, отказы YuNet и строка детекции), own_moments.tsv
 * (моменты своей базы: человек, момент, кадры с отметкой качества, представитель — по ним восстанавливаются
 * 12 ротаций), splits.tsv (Georgia Tech — три разбиения, MUCT — валидация и контроль; seed), MANIFEST.txt
 * (SHA-256 и размер каждого файла, число по базам), README.md (генерируемая часть между метками, ручной текст
 * сохраняется). Каталог reference/ экспорт не удаляет и не перезаписывает. Сырых снимков и моделей в экспорте нет.
 *
 * GalleryEvaluation при заданном faces.export.dir читает все четыре базы отсюда (read): векторы — из PNG,
 * отказы и разбиения — из файлов; разбиения пересчитываются по seed и сверяются со splits.tsv.
 *
 * @author ssv
 */
public final class FarExport {

    static final String DETECTIONS = "detections.tsv";
    static final String MOMENTS = "own_moments.tsv";
    static final String SPLITS = "splits.tsv";
    static final String MANIFEST = "MANIFEST.txt";
    static final String README = "README.md";
    static final String FEI_SELECTION = "fei_selection.tsv";
    static final String REFERENCE = "reference";

    private FarExport() {
    }

    public static void main(String[] args) throws IOException {
        nu.pattern.OpenCV.loadLocally();
        long start = System.nanoTime();
        SettingsStore settings = new SettingsStore();
        String exportDir = settings.loadFacesExportDir();
        if (exportDir == null) {
            throw new IllegalStateException("Не задан каталог экспорта: faces.export.dir (settings.properties) или FACES_EXPORT_DIR");
        }
        Path dir = Paths.get(exportDir);
        long seed = settings.loadFacesSeed();
        float score = settings.loadFacesOwnDetectorScore();
        Path modelsDir = Paths.get(settings.loadFacesModelsDir());
        Path outDir = Paths.get(System.getProperty("user.dir"), "reports", "faces", "far");
        Files.createDirectories(outDir);
        boolean fresh = args.length > 0 && args[0].equals("fresh");
        GalleryEvaluation.Raw raw = GalleryEvaluation.raw(settings, score, modelsDir, outDir, fresh);
        Data data = raw.data();
        Map<String, Det> dets = raw.dets();
        FaceRecognizerSF sface = FaceRecognizerSF.create(modelsDir.resolve(FaceAlignment.SFACE_FILE).toString(), "");

        // Каталог экспорта: прежние кадры и файлы экспорта удаляются; reference/, .git и прочее не трогается.
        // Ручная часть README.md (вне генерируемой, в том числе строки о reference/) сохраняется.
        String oldReadme = clean(dir);

        // FEI: список отбора; справочные наборы как есть (в протокол не входят) — только если reference/FEI ещё нет;
        // исходные снимки не копируются.
        if (raw.fei() != null) {
            write(dir.resolve(FEI_SELECTION), raw.fei().table());
            Path feiRef = dir.resolve(REFERENCE).resolve("FEI");
            if (!Files.exists(feiRef)) copyReference(Paths.get(settings.loadFacesFeiDir()), feiRef);
        }

        // Кадры и detections.tsv.
        StringBuilder det = new StringBuilder("base\tperson\tname\tcamera\tfound\trow (15 чисел YuNet, координаты полного снимка)\n");
        int done = 0;
        for (Sample s : data.samples()) {
            double[] row = dets.get(s.file().toString()).row();
            Path id = id(s);
            det.append(dirName(s.base())).append('\t').append(s.person()).append('\t').append(id.getFileName()).append('\t')
                    .append(s.camera() == ' ' ? "-" : String.valueOf(s.camera())).append('\t').append(row == null ? 0 : 1);
            if (row != null) {
                for (double x : row) det.append('\t').append(x);
                Mat[] m = GalleryEvaluation.load(s);
                try {
                    for (Variant v : Variant.values()) {
                        Mat frame = GalleryEvaluation.frame(s, m, row, v, sface);
                        writePng(png(dir, v, id), frame);
                        frame.release();
                    }
                } finally {
                    GalleryEvaluation.release(m);
                }
            }
            det.append('\n');
            if (++done % 250 == 0) System.out.println("Экспорт: " + done + "/" + data.samples().size());
        }
        write(dir.resolve(DETECTIONS), det);

        // Моменты своей базы: все моменты и кадры (включая непригодные) — по ним восстанавливается OwnDataset.
        StringBuilder mom = new StringBuilder("person\tsecond\tname\tindex\tquality\ttime\trepresentative\n");
        for (OwnDataset.Person p : data.own().persons()) {
            for (OwnDataset.Moment m : p.moments()) {
                OwnDataset.Frame rep = m.representative();
                for (OwnDataset.Frame f : m.frames()) {
                    mom.append(p.name()).append('\t').append(m.second()).append('\t').append(stem(f.file())).append('\t')
                            .append(f.index()).append('\t').append(f.quality()).append('\t').append(f.time()).append('\t')
                            .append(f == rep ? 1 : 0).append('\n');
                }
            }
        }
        write(dir.resolve(MOMENTS), mom);
        write(dir.resolve(SPLITS), splitsText(GalleryEvaluation.splits(data.gt(), data.muct(), seed), seed));
        write(dir.resolve(README), mergeReadme(oldReadme, readme(data, dets).toString()));
        write(dir.resolve(MANIFEST), manifest(dir, data, dets));
        System.out.printf(Locale.ROOT, "Экспорт в %s: %.0f с%n", dir, (System.nanoTime() - start) / 1e9);
    }

    // ---------------------------------------------------------------- имена

    static String dirName(Variant v) {
        return v == Variant.SFACE ? "sface" : v.name().toLowerCase(Locale.ROOT);
    }

    static String dirName(Base b) {
        return b.name().toLowerCase(Locale.ROOT);
    }

    /** Имя файла без расширения. */
    static String stem(Path file) {
        String n = file.getFileName().toString();
        int dot = n.lastIndexOf('.');
        return dot < 0 ? n : n.substring(0, dot);
    }

    /** Относительный путь снимка в экспорте без расширения: база/человек/имя. */
    static Path id(Sample s) {
        return Paths.get(dirName(s.base()), s.person(), stem(s.file()));
    }

    static Path png(Path dir, Variant v, Path id) {
        return dir.resolve(dirName(v)).resolve(id.toString() + ".png");
    }

    // ---------------------------------------------------------------- чтение (GalleryEvaluation)

    /** Экспорт в памяти: базы (снимки — относительные пути база/человек/имя), найденные лица, строки splits.tsv. */
    record Loaded(Path dir, Data data, Map<String, Boolean> found, List<String> splitLines, long seedInFile,
                  FeiDataset.Info feiInfo) {

        /** Разбиения, пересчитанные по seed, должны совпасть со splits.tsv. */
        void checkSplits(Splits splits) {
            List<String> expected = splitsText(splits, seedInFile).toString().lines().toList();
            if (!expected.equals(splitLines)) {
                throw new IllegalStateException("Разбиения, пересчитанные по seed " + seedInFile + ", не совпадают с " + dir.resolve(SPLITS));
            }
        }

        /** Кадр снимка варианта v из PNG без векторизации (FarMethods); отказ детектора — null. */
        Mat frame(Sample s, Variant v) throws IOException {
            String key = s.file().toString();
            return found.get(key) ? readPng(dir.resolve(dirName(v)).resolve(key + ".png"), v == Variant.SFACE) : null;
        }

        /** Векторы одного варианта из PNG (отказ детектора — Feat.NONE). */
        Map<String, Feat> vectors(List<Sample> samples, Variant v, FaceRecognizerSF sface) throws IOException {
            Map<String, Feat> feats = new LinkedHashMap<>();
            int done = 0;
            for (Sample s : samples) {
                String key = s.file().toString();
                if (!found.get(key)) {
                    feats.put(key, Feat.NONE);
                } else {
                    Mat frame = readPng(dir.resolve(dirName(v)).resolve(key + ".png"), v == Variant.SFACE);
                    double[] x = GalleryEvaluation.vector(frame, v, sface);
                    frame.release();
                    feats.put(key, new Feat(true, v == Variant.A ? x : null, v == Variant.B ? x : null, v == Variant.C ? x : null,
                            v == Variant.SFACE ? x : null));
                }
                if (++done % 1000 == 0) System.out.println(v.label + ": векторы " + done + "/" + samples.size());
            }
            return feats;
        }
    }

    /** Читает экспорт: detections.tsv, own_moments.tsv, splits.tsv; сырые снимки не нужны. */
    static Loaded read(Path dir) throws IOException {
        // Своя база — из own_moments.tsv (порядок людей, моментов и кадров — как в OwnDataset.load).
        Map<String, Map<Integer, List<OwnDataset.Frame>>> byPerson = new LinkedHashMap<>();
        Map<OwnDataset.Frame, Boolean> repFlag = new HashMap<>();
        for (String line : body(dir.resolve(MOMENTS))) {
            String[] c = line.split("\t", -1);
            OwnDataset.Frame f = new OwnDataset.Frame(Paths.get(dirName(Base.OWN), c[0], c[2]), Integer.parseInt(c[1]),
                    Integer.parseInt(c[3]), Integer.parseInt(c[4]), Double.parseDouble(c[5]));
            byPerson.computeIfAbsent(c[0], k -> new LinkedHashMap<>()).computeIfAbsent(f.second(), k -> new ArrayList<>()).add(f);
            repFlag.put(f, c[6].equals("1"));
        }
        List<OwnDataset.Person> persons = new ArrayList<>();
        byPerson.forEach((name, moments) -> {
            List<OwnDataset.Moment> list = new ArrayList<>();
            moments.forEach((second, frames) -> list.add(new OwnDataset.Moment(second, List.copyOf(frames))));
            persons.add(new OwnDataset.Person(name, List.copyOf(list)));
        });
        for (OwnDataset.Person p : persons) {
            for (OwnDataset.Moment m : p.moments()) {
                for (OwnDataset.Frame f : m.frames()) {
                    if (repFlag.get(f) != (f == m.representative())) {
                        throw new IllegalStateException("Представитель момента не совпадает с " + MOMENTS + ": " + f.file());
                    }
                }
            }
        }
        OwnDataset own = OwnDataset.of(dir, persons);

        // Остальные базы и отказы детектора — из detections.tsv.
        Map<String, Boolean> found = new HashMap<>();
        List<String> order = new ArrayList<>();
        Map<String, List<Sample>> gt = new TreeMap<>();
        Map<String, List<Sample>> muct = new TreeMap<>();
        Map<String, List<Sample>> orl = new TreeMap<>();
        Map<String, List<Sample>> fei = new TreeMap<>();
        for (String line : body(dir.resolve(DETECTIONS))) {
            String[] c = line.split("\t", -1);
            Base base = Base.valueOf(c[0].toUpperCase(Locale.ROOT));
            Path id = Paths.get(c[0], c[1], c[2]);
            found.put(id.toString(), c[4].equals("1"));
            order.add(id.toString());
            Sample s = new Sample(base, c[1], id, c[3].equals("-") ? ' ' : c[3].charAt(0));
            switch (base) {
                case GT -> gt.computeIfAbsent(c[1], k -> new ArrayList<>()).add(s);
                case MUCT -> muct.computeIfAbsent(c[1], k -> new ArrayList<>()).add(s);
                case ORL -> orl.computeIfAbsent(c[1], k -> new ArrayList<>()).add(s);
                case FEI -> fei.computeIfAbsent(c[1], k -> new ArrayList<>()).add(s);
                case OWN -> { }
            }
        }
        Data data = GalleryEvaluation.data(own, gt, muct, orl, fei);
        List<String> rebuilt = data.samples().stream().map(s -> s.file().toString()).toList();
        if (!rebuilt.equals(order)) {
            throw new IllegalStateException("Снимки, восстановленные по " + MOMENTS + " и " + DETECTIONS + ", не совпадают с порядком "
                    + DETECTIONS);
        }
        List<String> splitLines = Files.readAllLines(dir.resolve(SPLITS), StandardCharsets.UTF_8);
        long seed = Long.parseLong(splitLines.get(0).substring(splitLines.get(0).lastIndexOf(' ') + 1));
        System.out.println("Базы — из экспорта " + dir + " (снимков " + order.size() + ")");
        Path sel = dir.resolve(FEI_SELECTION);
        FeiDataset.Info feiInfo = fei.isEmpty() || !Files.exists(sel) ? null
                : FeiDataset.readInfo(Files.readAllLines(sel, StandardCharsets.UTF_8));
        if (!fei.isEmpty() && feiInfo == null) throw new IllegalStateException("В экспорте есть FEI, но нет " + sel);
        return new Loaded(dir, data, found, splitLines, seed, feiInfo);
    }

    /** Строки файла без заголовка. */
    private static List<String> body(Path file) throws IOException {
        List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
        return lines.subList(1, lines.size());
    }

    // ---------------------------------------------------------------- тексты

    /** splits.tsv: seed; GT — разбиение, человек, обучающие индексы (номера снимков человека с 0); MUCT — часть, человек. */
    static StringBuilder splitsText(Splits sp, long seed) {
        StringBuilder t = new StringBuilder("# seed " + seed + "\n");
        t.append("base\tpart\tperson\ttrain (GT: индексы снимков человека по порядку имён, с 0)\n");
        for (int s = 0; s < sp.gtTrain().size(); s++) {
            for (int p = 0; p < sp.gtPersons().size(); p++) {
                t.append("gt\t").append(s).append('\t').append(sp.gtPersons().get(p)).append('\t')
                        .append(sp.gtTrain().get(s).get(p).stream().map(String::valueOf).collect(Collectors.joining(","))).append('\n');
            }
        }
        for (String p : sp.muctVal()) t.append("muct\tval\t").append(p).append("\t\n");
        for (String p : sp.muctCtrl()) t.append("muct\tctrl\t").append(p).append("\t\n");
        return t;
    }

    /** Число снимков и найденных лиц по базам: {снимков, найдено}. */
    private static Map<Base, int[]> counts(Data data, Map<String, Det> dets) {
        Map<Base, int[]> c = new LinkedHashMap<>();
        for (Base b : Base.values()) c.put(b, new int[2]);
        for (Sample s : data.samples()) {
            c.get(s.base())[0]++;
            if (dets.get(s.file().toString()).row() != null) c.get(s.base())[1]++;
        }
        return c;
    }

    private static StringBuilder readme(Data data, Map<String, Det> dets) {
        Map<Base, int[]> c = counts(data, dets);
        StringBuilder t = new StringBuilder();
        t.append("# SVD-faces-data\n\n");
        t.append("Приватный репозиторий данных проекта SVD-ShapeRecognizer (распознавание лиц методом SVD, шаг 5): выровненные\n"
                + "кадры лиц для расчётов в облачной сессии. Сформирован командой `mvn compile exec:exec@faces-export`\n"
                + "(класс `svd.recognizer.faces.FarExport`). Не распространять.\n\n");
        t.append("## Состав\n\n");
        t.append("| База | Людей | Снимков | Лицо найдено (кадров на вариант) |\n|---|---|---|---|\n");
        t.append(String.format(Locale.ROOT, "| own (своя база) | %d | %d | %d |%n", data.own().persons().size(), c.get(Base.OWN)[0], c.get(Base.OWN)[1]));
        t.append(String.format(Locale.ROOT, "| gt (Georgia Tech) | %d | %d | %d |%n", data.gt().size(), c.get(Base.GT)[0], c.get(Base.GT)[1]));
        t.append(String.format(Locale.ROOT, "| muct (MUCT) | %d | %d | %d |%n", data.muct().size(), c.get(Base.MUCT)[0], c.get(Base.MUCT)[1]));
        t.append(String.format(Locale.ROOT, "| orl (ORL) | %d | %d | %d |%n", data.orl().size(), c.get(Base.ORL)[0], c.get(Base.ORL)[1]));
        if (!data.fei().isEmpty()) {
            t.append(String.format(Locale.ROOT, "| fei (FEI, отобранные исходные снимки) | %d | %d | %d |%n", data.fei().size(),
                    c.get(Base.FEI)[0], c.get(Base.FEI)[1]));
        }
        t.append('\n');
        t.append("## Структура\n\n");
        t.append("- `a/`, `b/`, `c/`, `sface/` — кадры PNG без потерь, путь `<вариант>/<база>/<человек>/<имя снимка>.png`:\n"
                + "  - `a` — 92×112, серый, пиксели входа первого прохода детектора (своя база ×0,25);\n"
                + "  - `b` — 92×112, серый, пиксели из полного разрешения;\n"
                + "  - `c` — 184×224, серый, пиксели из полного разрешения;\n"
                + "  - `sface` — вход SFace, alignCrop 112×112, BGR.\n"
                + "- `detections.tsv` — все снимки в порядке оценки: база, человек, имя, камера MUCT, найдено ли лицо\n"
                + "  (двухпроходный YuNet, порог 0,7), строка детекции (15 чисел, координаты полного снимка). Отказ детектора —\n"
                + "  `found` = 0, кадров у снимка нет.\n"
                + "- `own_moments.tsv` — своя база: человек, момент (секунда), кадры с пометкой качества (3 «+», 2 «+-»,\n"
                + "  1 «+--», 0 без пометки), время съёмки, представитель момента; по нему восстанавливаются 12 ротаций.\n"
                + "- `splits.tsv` — разбиения: Georgia Tech — три разбиения (seed + номер), обучающие индексы; MUCT —\n"
                + "  валидация и контроль по людям (seed).\n"
                + "- `MANIFEST.txt` — SHA-256 и размер каждого файла, число по базам.\n");
        if (!data.fei().isEmpty()) {
            t.append("- `fei_selection.tsv` — отбор FEI: все 2800 исходных снимков (человек, номер, найдено ли лицо, поза r,\n"
                    + "  справочный «a»/«b», включён ли в протокол и почему); строки «#» — порог позы R и соответствие «a»/«b».\n"
                    + "- `reference/FEI/` — справочные наборы FEI как есть (frontalimages_manuallyaligned, spatiallynormalized,\n"
                    + "  cropped_equalized, средние лица, разметка 46 точек). **В протокол не входят**: по manuallyaligned только\n"
                    + "  установлено, каким исходным снимкам соответствуют «a» и «b».\n");
        }
        t.append('\n');
        t.append("Сырых снимков и моделей здесь нет. Ни один файл не больше 100 МБ, Git LFS не используется.\n\n");
        t.append("## Происхождение и условия\n\n");
        t.append("Базы используются на условиях их первоисточников; этот репозиторий никаких прав на них не даёт\n"
                + "(поэтому LICENSE нет).\n\n");
        t.append("- **ORL** (The Database of Faces, AT&T Laboratories Cambridge; Olivetti Research Laboratory, 1992–1994):\n"
                + "  https://cam-orl.co.uk/facedatabase.html . Условие — ссылаться на AT&T Laboratories Cambridge.\n");
        t.append("- **Georgia Tech Face Database** (A. V. Nefian): http://www.anefian.com/research/face_reco.htm .\n"
                + "  Лицензия у первоисточника не указана.\n");
        t.append("- **MUCT** (S. Milborrow, J. Morkel, F. Nicolls, The MUCT Landmarked Face Database, PRASA 2010):\n"
                + "  http://www.milbo.org/muct/ , https://github.com/StephenMilborrow/muct . Условие первоисточника — не\n"
                + "  воспроизводить снимки MUCT в публично доступных документах (кроме людей 000, 001, 002, 200, 201, 400, 401,\n"
                + "  402 — в академических статьях); при использовании цитировать Milborrow et al., 2010.\n");
        if (!data.fei().isEmpty()) {
            t.append("- **FEI Face Database** (Centro Universitário da FEI, São Bernardo do Campo, Бразилия):\n"
                    + "  https://fei.edu.br/~cet/facedatabase.html . Условия — только исследовательские цели; ссылаться на\n"
                    + "  C. E. Thomaz, G. A. Giraldi, A new ranking method for Principal Components Analysis and its application to\n"
                    + "  face image analysis, Image and Vision Computing 28(6), 902–913, 2010. Исходные снимки FEI (originalimages)\n"
                    + "  сюда не кладутся — только выровненные кадры отобранных снимков и справочные наборы в `reference/FEI`.\n");
        }
        t.append("- **Своя база** — снимки получены с согласия снятых людей только для этого проекта. Не распространять, не\n"
                + "  публиковать, не передавать третьим лицам. Здесь только выровненные кадры; исходные фото сюда не кладутся.\n\n");
        t.append("## Как подключить\n\n");
        t.append("```\n"
                + "git clone <адрес SVD-faces-data> /путь/к/данным\n"
                + "export FACES_EXPORT_DIR=/путь/к/данным          # или ключ faces.export.dir в settings.properties\n"
                + "export FACES_MODELS_DIR=/путь/к/моделям         # SFace нужен для признаков (face_recognition_sface_2021dec.onnx)\n"
                + "mvn compile exec:exec@faces-far\n"
                + "```\n\n"
                + "При заданном `faces.export.dir` (`FACES_EXPORT_DIR`) оценка `GalleryEvaluation` читает все четыре базы отсюда:\n"
                + "векторы — из PNG, отказы детектора и разбиения — из файлов (разбиения пересчитываются по seed и сверяются\n"
                + "со `splits.tsv`); сырые снимки не читаются.\n");
        return t;
    }

    private static StringBuilder manifest(Path dir, Data data, Map<String, Det> dets) throws IOException {
        List<Path> files;
        try (Stream<Path> s = Files.walk(dir)) {
            files = s.filter(Files::isRegularFile).filter(p -> !dir.relativize(p).startsWith(".git"))
                    .filter(p -> !p.getFileName().toString().equals(MANIFEST))
                    .sorted(Comparator.comparing(p -> dir.relativize(p).toString().replace('\\', '/'))).toList();
        }
        Map<String, long[]> byDir = new TreeMap<>();
        StringBuilder list = new StringBuilder();
        long total = 0;
        for (Path f : files) {
            String rel = dir.relativize(f).toString().replace('\\', '/');
            long size = Files.size(f);
            total += size;
            list.append(sha256(f)).append("  ").append(size).append("  ").append(rel).append('\n');
            String[] parts = rel.split("/");
            String group = parts.length >= 3 ? parts[0] + "/" + parts[1] : "(корень)";
            long[] g = byDir.computeIfAbsent(group, k -> new long[2]);
            g[0]++;
            g[1] += size;
        }
        Map<Base, int[]> c = counts(data, dets);
        StringBuilder t = new StringBuilder("MANIFEST SVD-faces-data (FarExport)\n\n");
        t.append("Снимков по базам (найдено лицо / всего): ");
        // FEI — только если подключена (без неё строка прежняя).
        List<String> parts = new ArrayList<>();
        for (Base b : Base.values()) {
            if (b == Base.FEI && c.get(b)[0] == 0) continue;
            parts.add(dirName(b) + ' ' + c.get(b)[1] + '/' + c.get(b)[0]);
        }
        t.append(String.join(", ", parts)).append('\n');
        t.append("Файлов и байт по каталогам (вариант/база):\n");
        byDir.forEach((k, v) -> t.append(String.format(Locale.ROOT, "  %-14s %6d файлов %,12d байт%n", k, v[0], v[1])));
        t.append(String.format(Locale.ROOT, "Всего (без MANIFEST.txt): %d файлов, %,d байт%n%n", files.size(), total));
        t.append("SHA-256  размер  путь\n").append(list);
        return t;
    }

    // ---------------------------------------------------------------- файлы

    static void writePng(Path file, Mat frame) throws IOException {
        MatOfByte buf = new MatOfByte();
        if (!Imgcodecs.imencode(".png", frame, buf)) throw new IOException("Не удалось закодировать PNG: " + file);
        Files.createDirectories(file.getParent());
        Files.write(file, buf.toArray());
        buf.release();
    }

    static Mat readPng(Path file, boolean color) throws IOException {
        Mat m = Imgcodecs.imdecode(new MatOfByte(Files.readAllBytes(file)), color ? Imgcodecs.IMREAD_COLOR : Imgcodecs.IMREAD_GRAYSCALE);
        if (m.empty()) throw new IOException("Не удалось декодировать " + file);
        return m;
    }

    private static void write(Path file, CharSequence text) throws IOException {
        Files.writeString(file, text, StandardCharsets.UTF_8);
    }

    /** Начало и конец генерируемой части README.md; текст вне них — ручной, при экспорте сохраняется. */
    static final String README_BEGIN = "<!-- FarExport: начало генерируемой части -->";
    static final String README_END = "<!-- FarExport: конец генерируемой части -->";

    /**
     * Очистка каталога экспорта перед записью: удаляются кадры вариантов и файлы экспорта; каталог reference/,
     * .git и прочие файлы не трогаются.
     *
     * @return прежний README.md (нет — null)
     */
    static String clean(Path dir) throws IOException {
        Files.createDirectories(dir);
        Path readme = dir.resolve(README);
        String old = Files.exists(readme) ? Files.readString(readme, StandardCharsets.UTF_8) : null;
        for (Variant v : Variant.values()) deleteTree(dir.resolve(dirName(v)));
        for (String f : new String[] {DETECTIONS, MOMENTS, SPLITS, FEI_SELECTION, MANIFEST, README}) Files.deleteIfExists(dir.resolve(f));
        return old;
    }

    /**
     * Новый README.md: генерируемая часть между метками, ручной текст прежнего README — как был. Прежний README с
     * метками: заменяется только часть между ними. Без меток (README до этой правки): ручными считаются абзацы, где
     * упомянут reference/ и которых нет в генерируемой части, — они идут после неё.
     */
    static String mergeReadme(String old, String generated) {
        String block = README_BEGIN + "\n" + generated + (generated.endsWith("\n") ? "" : "\n") + README_END + "\n";
        if (old == null) return block;
        int b = old.indexOf(README_BEGIN);
        int e = old.indexOf(README_END);
        if (b >= 0 && e > b) {
            int after = e + README_END.length();
            if (old.startsWith("\r\n", after)) after += 2;
            else if (old.startsWith("\n", after)) after += 1;
            return old.substring(0, b) + block + old.substring(after);
        }
        StringBuilder kept = new StringBuilder();
        for (String par : old.replace("\r\n", "\n").split("\n\n+")) {
            String p = par.strip();
            if (p.contains(REFERENCE + "/") && !generated.contains(p)) kept.append('\n').append(p).append('\n');
        }
        return block + kept;
    }

    /** Справочные наборы FEI как есть: всё, кроме originalimages и SOURCE.txt. */
    private static void copyReference(Path fei, Path target) throws IOException {
        List<Path> files;
        try (Stream<Path> s = Files.walk(fei)) {
            files = s.filter(Files::isRegularFile).filter(p -> !fei.relativize(p).startsWith(FeiDataset.ORIGINALS))
                    .filter(p -> !fei.relativize(p).toString().equals("SOURCE.txt")).sorted().toList();
        }
        for (Path f : files) {
            Path t = target.resolve(fei.relativize(f).toString());
            Files.createDirectories(t.getParent());
            Files.copy(f, t);
        }
    }

    private static void deleteTree(Path root) throws IOException {
        if (!Files.exists(root)) return;
        List<Path> all;
        try (Stream<Path> s = Files.walk(root)) {
            all = s.sorted(Comparator.reverseOrder()).toList();
        }
        for (Path p : all) Files.delete(p);
    }

    private static String sha256(Path file) throws IOException {
        try (InputStream in = Files.newInputStream(file)) {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] buf = new byte[1 << 16];
            int n;
            while ((n = in.read(buf)) > 0) md.update(buf, 0, n);
            return HexFormat.of().formatHex(md.digest());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }
}
