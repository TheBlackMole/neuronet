import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Random;

public class Network {

    private NeuronLayer hiddenLayer;
    private NeuronLayer outputLayer;

    public Network(int inputLayerSize, int hiddenLayerSize, int outputLayerSize) {
        hiddenLayer = new NeuronLayer(hiddenLayerSize, inputLayerSize);
        outputLayer = new NeuronLayer(outputLayerSize, hiddenLayerSize);
    }

    public double[] run(double[] input) {
        System.out.println("Inputs: " + input [0] + " " + input[1]);

        double[] resultsHiddenLayer = hiddenLayer.forward(input);
        return outputLayer.forward(resultsHiddenLayer);
    }

    public void training(int trainingSize, List<double[]> trainingData) {
        Random r = new Random();


        for(int i = 1; i < trainingSize; i++) {
            // Test durchführen und Kosten berechnen
            double[] currentData = trainingData.get(r.nextInt(trainingData.size()));
            double[] input = {currentData[0], currentData[1]};
            double[] results = run(input);
            double[] trainingDataResults = {currentData[2]};
            double cost = calculateCostFunction(results, trainingDataResults);

            // Werte anpassen

            NeuronLayer oldHiddenLayer = hiddenLayer.clone();
            NeuronLayer oldOutputLayer = outputLayer.clone();
            hiddenLayer.changeSingleValue(cost);
            outputLayer.changeSingleValue(cost);

            // Neu berechnen
            results = run(input);
            double newCost = calculateCostFunction(results, trainingDataResults);
            System.out.println("Alte Kosten: " + cost + " Neue Kosten: " + newCost);

            // Kostenwerte notieren zum Plotten
            // Writer erstellen
            String filename = "kosten.txt";
            try (FileWriter writer = new FileWriter(filename, true)) {
                writer.write(Double.toString(cost) + "\n");
            } catch (IOException e) {
                e.printStackTrace();
            }



            if(newCost > cost) { // Bei Verschlechterung rückgängig
                hiddenLayer = oldHiddenLayer;
                outputLayer = oldOutputLayer;
                System.out.println("Wird nicht übernommen");
                continue;
            }
            System.out.println("Wird übernommen");


        }
    }

    private double calculateCostFunction(double[] results, double[] trainingDataResults) {
        double cost = 0;
        for(int i = 0; i < results.length && i < trainingDataResults.length; i++) {
            cost += Math.pow((trainingDataResults[i] - results[i]), 2);
        }
        return cost;
    }
}
