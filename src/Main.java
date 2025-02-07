import java.util.List;

public class Main {


    public static void main(String[] args) {
        Data data = new Data();
        data.createTrainingData(10000);
        Network network = new Network(2, 1, 1);
        double[] input = {1,2};
        double[] results = network.run(input);
        System.out.println("Ausgabe:");
        for(double result : results) {
            System.out.println(result);
        }

        List<double[]> trainingData = Data.loadTrainingData("trainingsdaten.txt");
        for (int i = 0; i < 5 && i < trainingData.size(); i++) {
            double[] d = trainingData.get(i);
            System.out.println("x: " + d[0] + ", y: " + d[1] + ", label: " + d[2]);
        }
    }

}
