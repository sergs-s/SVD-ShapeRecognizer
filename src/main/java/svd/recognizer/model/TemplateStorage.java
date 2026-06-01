package svd.recognizer.model;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author ssv
 */
public class TemplateStorage {

    private static final String TEMPLATES_DIR = "templates";

    /**
     * Saves a ShapeTemplate to a .dat file in the templates directory.
     *
     * @param template the template to save
     * @throws IOException on file error
     */
    public static void save(ShapeTemplate template) throws IOException {
        File dir = new File(TEMPLATES_DIR);
        if (!dir.exists()) dir.mkdirs();
        File file = new File(dir, template.getLabel() + ".dat");
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
            oos.writeObject(template);
        }
    }

    /**
     * Loads all ShapeTemplates from the templates directory.
     *
     * @return list of loaded templates
     */
    public static List<ShapeTemplate> loadAll() {
        List<ShapeTemplate> list = new ArrayList<>();
        File dir = new File(TEMPLATES_DIR);
        if (!dir.exists()) return list;
        File[] files = dir.listFiles((d, name) -> name.endsWith(".dat"));
        if (files == null) return list;
        for (File f : files) {
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(f))) {
                list.add((ShapeTemplate) ois.readObject());
            } catch (Exception e) {
                System.err.println("Cannot load template: " + f.getName() + " - " + e.getMessage());
            }
        }
        return list;
    }

    /**
     * Deletes a template file by label.
     *
     * @param label shape label
     * @return true if deleted successfully
     */
    public static boolean delete(String label) {
        File file = new File(TEMPLATES_DIR, label + ".dat");
        return file.exists() && file.delete();
    }

    /**
     * Deletes all template files.
     */
    public static void deleteAll() {
        File dir = new File(TEMPLATES_DIR);
        if (!dir.exists()) return;
        File[] files = dir.listFiles((d, name) -> name.endsWith(".dat"));
        if (files == null) return;
        for (File f : files) f.delete();
    }
}
