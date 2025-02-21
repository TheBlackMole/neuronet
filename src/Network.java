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

        new GUI(this);
    }

    public double[] run(double[] input, String type) {
        //System.out.println("Inputs: " + input [0] + " " + input[1]);

        double[] resultsHiddenLayer = hiddenLayer.forward(input, type);
        return outputLayer.forward(resultsHiddenLayer, type);
    }

    public void training(int trainingSize, List<double[]> trainingData) {
        //Random r = new Random();

        double oldCost;
        double newCost;


        for(int i = 0; i < trainingSize; i++) {
            
            // Alte Kosten berechnen
            oldCost = calculateCurrentCostSum(trainingData, null);

            // Altes Netz sichern
            saveWeightsAndBiases("weights_biases.txt");

            // Layer anpassen
            hiddenLayer.changeValue(oldCost); // Hiddenlayer wird geändert
            outputLayer.changeValue(oldCost); // Outputlayer wird geändert
            System.out.println("Fehler vorher: " + calculateCurrentCostSum(trainingData, null));

            System.out.println("Activation vorher hidden: " + hiddenLayer.getActivationFunction());
            System.out.println("Avtivation vorher out: " + outputLayer.getActivationFunction());
            changeActivationFunktion(hiddenLayer, outputLayer, trainingData);
            System.out.println("Activation nacher hidden: " + hiddenLayer.getActivationFunction());
            System.out.println("Avtivation nacher out: " + outputLayer.getActivationFunction());

            System.out.println("Fehler nacher: " + calculateCurrentCostSum(trainingData, null) + "\n");
            // Neue Kosten berechnen
            newCost = calculateCurrentCostSum(trainingData, null);

            if (oldCost < newCost) { // Wenn Verschlechterung
                loadWeightsAndBiases("weights_biases.txt");
                costs.add(oldCost);
            } else { // Wenn Verbesserung
                saveWeightsAndBiases("weights_biases.txt");
                costs.add(newCost);
            }
            

        }

        printCost();

    }

    private void changeActivationFunktion(NeuronLayer hidden, NeuronLayer output, List<double[]> trainingData) {
        //Kosten mit allen Möglichkeiten der Aktivierungsfunktionen ausrechen und die Kosten in Array speichern
        Double[] costs = new Double[16];
        String[] activationF= new String[]{"sigmoid","tanh", "leakyReLu", "linear" };
        int stelle=0;
        for(int i=0; i<=3; i++) {
            hiddenLayer.setActivationFunction(activationF[i]);
            for(int y = 0; y<=3; y++){
                outputLayer.setActivationFunction(activationF[y]);
                costs[stelle]=calculateCurrentCostSum(trainingData, null);
                stelle++;
            }
        }
        //kleinste Kosten finden
        double minValue=costs[0];
        int minIndex = 0;
        for(int i = 1; i<=15; i++){
            if (costs[i] < minValue) {
                minValue=costs[i];
                minIndex = i;
            }
        }
        minIndex++;
        //kleinsten Aktivierungen setzen
        if (minIndex == 16) {
            hiddenLayer.setActivationFunction(activationF[3]);
            outputLayer.setActivationFunction(activationF[3]);
        } else {
            int hiddenStelle = minIndex / 4 +1;
            int outputStelle = minIndex - ((hiddenStelle-1)*4);
            hiddenLayer.setActivationFunction(activationF[hiddenStelle-1]);
            outputLayer.setActivationFunction(activationF[outputStelle]);
        }
        
    }

    // Berechnet die summierten Kosten aller Trainingsdaten
    private double calculateCurrentCostSum(List<double[]> trainingData, String activeType) {
        double cost = 0;
        double[] input = new double[inputLayerSize];
        double[] trainingResult = new double[outputLayerSize];
        double[] networkResult;
        for(int n = 0; n < trainingData.size(); n++) {
            input[0] = trainingData.get(n)[0];
            input[1] = trainingData.get(n)[1];
            trainingResult[0] = trainingData.get(n)[2];
            networkResult = run(input, activeType);
            /*
            System.out.println("Input: " + input[0] + ", " + input[1]);
            System.out.println("trainingResult: " + trainingResult[0]);
            System.out.println("Ergebnis: " + networkResult[0]);
             */
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

    public static String findMinVariable(double a, double b, double c, double d) {
        double minValue = a;
        String minVariable = "sigmoid";

        if (b < minValue) {
            minValue = b;
            minVariable = "tanh";
        }
        if (c < minValue) {
            minValue = c;
            minVariable = "leakyReLu";
        }
        if (d < minValue) {
            minValue = d;
            minVariable = "linear";
        }

        return minVariable;
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

    public NeuronLayer getHiddenLayer() {
        return hiddenLayer;
    }

    public NeuronLayer getOutputLayer() {
        return outputLayer;
    }
}
