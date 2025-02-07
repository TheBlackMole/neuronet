import java.util.List;

public class Main {


    public static void main(String[] args) {
        Data data = new Data();
        data.createTrainingData(1000);
        List<double[]> trainingData = Data.loadTrainingData("trainingsdaten.txt");
        Network network = new Network(2, 1, 1);
        network.training(1000, trainingData);
        double[] input = {5, 10};
        double[] results = network.run(input);
        System.out.println("Ausgabe:");
        for(double result : results) {
            System.out.println(result);
        }


    }

}
