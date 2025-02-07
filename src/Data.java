import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Data {

    public Data() {

    }

    public void createTrainingData(int dataSize) {
        String filename = "trainingsdaten.txt";
        Random rand = new Random();

        try (FileWriter writer = new FileWriter(filename)) {
            writer.write("x,y,label\n"); // Header

            for (int i = 0; i < dataSize; i++) {
                int x = rand.nextInt(5000) + 1;  // Zufallszahl 1-500
                boolean isDouble = rand.nextBoolean();
                int y = isDouble ? x * 2 : rand.nextInt(10000) + 1; // Doppelt oder Zufall
                double label = isDouble ? 1.0 : 0.0;

                writer.write(x + "," + y + "," + label + "\n");
            }

            System.out.println("1000 Trainingsdaten wurden in " + filename + " gespeichert.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static List<double[]> loadTrainingData(String filename) {
        List<double[]> data = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            br.readLine(); // Erste Zeile (Header) überspringen

            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                double x = Double.parseDouble(parts[0]);
                double y = Double.parseDouble(parts[1]);
                double label = Double.parseDouble(parts[2]);
                data.add(new double[]{x, y, label});
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return data;
    }
}
