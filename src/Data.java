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
            writer.write("x,y,label1,label2\n"); // Header

            for (int i = 0; i < dataSize; i++) {
                int x = rand.nextInt(50) + 1;  // Zufallszahl 1-50
                int y = rand.nextInt(50) + 1; // Zufallszahl 1-50
                double label1 = gaussCurve(x, y);

                writer.write(x + "," + y + "," + label1 + "\n");
            }

            System.out.println(dataSize + " Trainingsdaten wurden in " + filename + " gespeichert.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static double gaussCurve(double x, double y) {
        // return 2 * Math.exp(-0.01 * Math.pow((x-y), 2)) - 1; // Für tanh Werte
        return Math.exp(-0.01 * Math.pow((x-y), 2)); // Für Sigmoid Werte
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
