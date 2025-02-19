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

    public void createTrainingData2(int dataSize) {
        Random random = new Random();
        StringBuilder data = new StringBuilder("Input1, Input2, Output\n");

        for (int i = 0; i < dataSize; i++) {
            int input1 = random.nextInt(10) + 1;
            int input2 = random.nextInt(10) + 1;
            double output = calculateOutput(input1, input2);
            data.append(input1).append(", ").append(input2).append(", ").append(output).append("\n");
        }

        try (FileWriter fileWriter = new FileWriter("training_data.csv")) {
            fileWriter.write(data.toString());
            System.out.println(dataSize + " Trainingsdatensätze wurden in 'training_data.csv' gespeichert.");
        } catch (IOException e) {
            System.err.println("Fehler beim Schreiben der Datei: " + e.getMessage());
        }
    }

    public static double calculateOutput(int input1, int input2) {
        if (input1 == input2) {
            return 0;
        } else if (input1 > input2) {
            return -1.0 * Math.abs(input1 - input2) / 10;
        } else {
            return Math.abs(input1 - input2) / 10;
        }
    }
    

    public void createTrainingData(int dataSize) {
        String filename = "trainingsdaten.txt";
        Random rand = new Random();

        try (FileWriter writer = new FileWriter(filename)) {
            writer.write("x,y,label1,label2\n"); // Header

            for (int i = 0; i < dataSize; i++) {
                int x = rand.nextInt(100) + 1;  // Zufallszahl 1-50
                int y = rand.nextInt(100) + 1; // Zufallszahl 1-50
                double label1 = Neuron.roundDouble(gaussCurve(x,y), 4);

                writer.write(Math.min(x,y) + "," + Math.max(x,y) + "," + label1 + "\n");
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
