import java.util.List;
import java.util.Random;

public class Main {


    public static void main(String[] args) {

        Data data = new Data();
        data.createTrainingData(5000);
        List<double[]> trainingData = Data.loadTrainingData("trainingsdaten.txt");
        Network network = new Network(2, 4, 1);
        //network.training(1000, trainingData);
        double[] input = {10, 10};
        double[] results = network.run(input);
        System.out.println("Ausgabe des Input:");
        for(double result : results) {
            System.out.println(result);
        }
        System.out.println("\n\n\n");
        Random r = new Random();
        for(int i = 0; i < 10; i++) { // Verschieden In- und Outputs testen -> aktuell führen ALLE zum selben Ergebnis
            int x = r.nextInt(100) + 1;
            int y = r.nextInt(100) + 1;
            //input[1] = input[0] + 20;
            input[0] = Math.min(x,y);
            input[1] = Math.max(x,y);
            results = network.run(input);
            System.out.println(results[0]);
        }


    }

}
