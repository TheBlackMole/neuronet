import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Main {


    public static void main(String[] args) {

        Data data = new Data();
        data.createTrainingData2(1000);
        List<double[]> trainingData = Data.loadTrainingData("training_data.csv");
        Network network = new Network(2, 4, 1);
        network.training(1000, trainingData);
        double[] input = {10, 10};
        double[] results = network.run(input);
        System.out.println("Input " + Arrays.toString(input));
        for(double result : results) {
            System.out.println(Neuron.roundDouble(result,3));
        }
        System.out.println("\n\n\n");
        Random r = new Random();
        for(int i = 0; i < 50; i++) { // Verschieden In- und Outputs testen -> aktuell führen ALLE zum selben Ergebnis
            int x = r.nextInt(100) + 1;
            //int y = r.nextInt(100) + 1;
            int y = x - i;
            input[0] = x;
            input[1] = y;
            results = network.run(input);
            System.out.println("Input " + Arrays.toString(input));
            System.out.println(Neuron.roundDouble(results[0],3));
        }


    }

}
