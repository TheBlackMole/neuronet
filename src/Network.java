//import java.io.FileWriter;
//import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;



public class Network {

    private NeuronLayer hiddenLayer;
    private NeuronLayer outputLayer;
    int inputLayerSize;
    int hiddenLayerSize;
    int outputLayerSize;
    List<Double> costs;

    public Network(int inputLayerSize, int hiddenLayerSize, int outputLayerSize) {
        this.inputLayerSize = inputLayerSize;
        this.hiddenLayerSize = hiddenLayerSize;
        this.outputLayerSize = outputLayerSize;
        hiddenLayer = new NeuronLayer(hiddenLayerSize, inputLayerSize);
        outputLayer = new NeuronLayer(outputLayerSize, hiddenLayerSize);
        loadWeightsAndBiases("weights_biases.txt");
        costs = new ArrayList<Double>();

        new GUI(hiddenLayer, outputLayer);
    }

    public double[] run(double[] input) {
        //System.out.println("Inputs: " + input [0] + " " + input[1]);

        double[] resultsHiddenLayer = hiddenLayer.forward(input, "leakyReLu");
        return outputLayer.forward(resultsHiddenLayer, "sigmoid");
    }

    public void training(int trainingSize, List<double[]> trainingData) {
        Random r = new Random();
        //double BiasOutStart = outputLayer.getNeuron(0).getBias();
        //double BiasHiddenStart = hiddenLayer.getNeuron(0).getBias();

        double oldCost;
        double newCost;

        for(int i = 0; i < trainingSize; i++) {
            //System.out.println("Bias out start: " + roundDouble(BiasOutStart, 3));
            //System.out.println("Bias hidden start: " + roundDouble(BiasHiddenStart, 3));
            // Testdaten mit altem Netz durchführen und Kosten berechnen
            //System.out.println("HL. n. : " + hiddenLayer.getNeuron(0));
            //double[] currentData = trainingData.get(r.nextInt(trainingData.size())); // Trainieren per Zufall

            // Alte Kosten berechnen
            oldCost = calculateCurrentCostSum(trainingData);

            // Altes Netz sichern
            saveWeightsAndBiases("weights_biases.txt");

            // Layer anpassen
            hiddenLayer.changeValue(oldCost); // Hiddenlayer wird geändert
            outputLayer.changeValue(oldCost); // Outputlayer wird geändert

            // Neue Kosten berechnen
            newCost = calculateCurrentCostSum(trainingData);
            //double newResult = results[0];
                //double oldDelta = trainingDataResults[0] - oldResult;
                //double newDelta = trainingDataResults[0] - newResult;
                //System.out.println("oldresult: " + oldResult + " newresult: " + newResult);
                //System.out.println("oldDelta: " + oldDelta + " newDelta: " + newDelta);
                //System.out.println("trainingResult: " + trainingDataResults[0])
                //double hiddenBias = hiddenLayer.getNeuron(0).getBias();
                //System.out.println("hiddenLayer o. Neuron Bias: " + roundDouble(hiddenBias,3));
                //System.out.println("outputLayer o. Neuron Bias: " + roundDouble(outputLayer.getNeuron(0).getBias() , 3) );;
            if (oldCost < newCost) { // Wenn Verschlechterung
                loadWeightsAndBiases("weights_biases.txt");
                costs.add(oldCost);
            } else {
                saveWeightsAndBiases("weights_biases.txt");
                costs.add(newCost);
            }



        }

        printCost();
            
            
        //hiddenLayer.changeSingleValue(cost);
            //outputLayer.changeSingleValue(cost);

        /*
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

            */

        //System.out.println("Neoron Out.0 Bias ende: " + roundDouble(outputLayer.getNeuron(0).getBias(), 3));
        //System.out.println("Neuron hidden.0 Bias ende: " + roundDouble(hiddenLayer.getNeuron(0).getBias(), 3));
    }

    // Berechnet die summierten Kosten aller Trainingsdaten
    private double calculateCurrentCostSum(List<double[]> trainingData) {
        double cost = 0;
        double[] input = new double[inputLayerSize];
        double[] trainingResult = new double[outputLayerSize];
        double[] networkResult;
        for(int n = 0; n < trainingData.size(); n++) {
            input[0] = trainingData.get(n)[0];
            input[1] = trainingData.get(n)[1];
            trainingResult[0] = trainingData.get(n)[2];
            networkResult = run(input);
            cost += calculateCostFunction(networkResult, trainingResult);
        }
        return cost;
    }

    public void saveWeightsAndBiases(String filename) {
        try (FileWriter file = new FileWriter(filename)) {
            // Speichern der Gewichte und Biases der versteckten Schicht
            for (Neuron neuron : hiddenLayer.getNeurons()) {
                file.write(roundDouble(neuron.getBias(), 4) + "\n");
                file.write(Arrays.toString(neuron.getWeights()) + "\n");
            }
            
            // Speichern der Gewichte und Biases der Ausgabeschicht
            for (Neuron neuron : outputLayer.getNeurons()) {
                file.write(neuron.getBias() + "\n");
                file.write(Arrays.toString(neuron.getWeights()) + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void loadWeightsAndBiases(String filename) {
    try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
        String line;
        
        // Laden der Gewichte und Biases der versteckten Schicht
        for (Neuron neuron : hiddenLayer.getNeurons()) {
            if ((line = reader.readLine()) != null) {
                neuron.setBias(Double.parseDouble(line));
            }
            if ((line = reader.readLine()) != null) {
                double[] weights = Arrays.stream(line.replace("[", "").replace("]", "").split(","))
                                         .mapToDouble(Double::parseDouble).toArray();
                neuron.setWeights(weights);
            }
        }

        // Laden der Gewichte und Biases der Ausgabeschicht
        for (Neuron neuron : outputLayer.getNeurons()) {
            if ((line = reader.readLine()) != null) {
                neuron.setBias(Double.parseDouble(line));
            }
            if ((line = reader.readLine()) != null) {
                double[] weights = Arrays.stream(line.replace("[", "").replace("]", "").split(","))
                                         .mapToDouble(Double::parseDouble).toArray();
                neuron.setWeights(weights);
            }
        }
    } catch (IOException e) {
        e.printStackTrace();
    }
}

    private double roundDouble(double value, int places) {
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

    // Kosten in Datei speichern
    private void printCost() {
        try (FileWriter file = new FileWriter("kosten.txt")) {
            for(Double d : costs) {
                file.write(Double.toString(roundDouble(d, 4)) + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
