import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
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
            System.out.println("HL. n. : " + hiddenLayer.getNeuron(0));
            double[] currentData = trainingData.get(r.nextInt(trainingData.size()));
            double[] input = {currentData[0], currentData[1]};
            double[] results = run(input);
            double oldResult = results[0];
            System.out.println("result: " + results[0]); 
            double[] trainingDataResults = {currentData[2]};
            System.out.println("trainingResult: " + trainingDataResults[0]);
            double cost = calculateCostFunction(results, trainingDataResults);

            // Werte anpassen

            NeuronLayer oldHiddenLayer = hiddenLayer.getCopy();
            NeuronLayer oldOutputLayer = outputLayer.getCopy();
            boolean goOn = true;
            int barrier = 0;
            while (goOn == true) {
                barrier++;
                System.out.println("barrier: " + barrier);
                outputLayer.changeValue(results[0], trainingDataResults[0]);
                results = run(input);
                double newResult = results[0];
                double oldDelta = trainingDataResults[0] - oldResult;
                double newDelta = trainingDataResults[0] - newResult;
                //System.out.println("oldresult: " + oldResult + " newresult: " + newResult);
                //System.out.println("oldDelta: " + oldDelta + " newDelta: " + newDelta);
                //System.out.println("trainingResult: " + trainingDataResults[0])
                double hiddenBias = hiddenLayer.getNeuron(0).getBias();
                System.out.println("hiddenLayer o. Neuron Bias: " + roundDouble(hiddenBias,3));
                System.out.println("outputLayer o. Neuron Bias: " + roundDouble(outputLayer.getNeuron(0).getBias() , 3) );;
                if (oldDelta > newDelta && newDelta != 0) {
                    hiddenLayer.changeValue(newResult, trainingDataResults[0]);
                }
                
                if(barrier>=10 || newDelta == 0){
                    goOn = false;
                    break;
                }
            }
            
            //hiddenLayer.changeSingleValue(cost);
            //outputLayer.changeSingleValue(cost);

            // Neu berechnen
            results = run(input);
            double newCost = calculateCostFunction(results, trainingDataResults);
            System.out.println("Alte Kosten: " + cost + " Neue Kosten: " + newCost);

            // Kostenwerte notieren zum Plotten
            // Writer erstellen
            String filename = "kosten.txt";
            try (FileWriter writer = new FileWriter(filename, true)) {
                writer.write(cost + "\n");
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

    public double roundDouble(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();

        BigDecimal bd = new BigDecimal(Double.toString(value));
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }

    private double calculateCostFunction(double[] results, double[] trainingDataResults) {
        double cost = 0;
        for(int i = 0; i < results.length && i < trainingDataResults.length; i++) {
            cost += Math.pow((trainingDataResults[i] - results[i]), 2);
        }
        return cost;
    }
}
