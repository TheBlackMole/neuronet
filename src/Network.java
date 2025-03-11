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
    int inputLayerSize;
    int outputLayerSize;
    List<Double> costs;
    public static List<double[]> trainingData = null;
    private GUI gui;
    private List<Integer> hiddenLayerSizes;
    private List<NeuronLayer> hiddenLayers;
    private NeuronLayer outputLayer;


    public Network(int inputLayerSize, List<Integer> hiddenLayerSizes, int outputLayerSize, int trainingDataSize) {
        this.inputLayerSize = inputLayerSize;
        this.outputLayerSize = outputLayerSize;
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.hiddenLayers = new ArrayList<>();
        initializeNetwork();

        try {
            loadWeightsAndBiases("weights_biases.txt");
        }
        catch (Exception e) {
            System.err.println("Die Datei 'weights_biases.txt' konnte nicht geladen werden.");
        }
        costs = new ArrayList<Double>();
        Data.createTrainingData2(trainingDataSize);
        trainingData = Data.loadTrainingData("training_data.csv");

        gui = new GUI(this);
    }

    private void initializeNetwork(){
        //NeuronLayer inputLayer = new NeuronLayer(inputLayerSize, 0);
        //layers.add(inputLayer);
        
        int previousLayerSize = inputLayerSize;
        for (int hiddenLayerSize : hiddenLayerSizes) {
            NeuronLayer hiddenLayer = new NeuronLayer(hiddenLayerSize, previousLayerSize);
            hiddenLayers.add(hiddenLayer);
            previousLayerSize = hiddenLayerSize;
        }

       outputLayer = new NeuronLayer(outputLayerSize, previousLayerSize);

    }

    

    public double[] run(double[] input) {
        //System.out.println("Inputs: " + input [0] + " " + input[1]);
        for (NeuronLayer layer : hiddenLayers) {
            input = layer.forward(input, null);
        }
        return outputLayer.forward(input, ActivationFunction.TANH);
    }

    
    public void preTraining() {
        double oldCost = 0;
        double newCost = 0;
        double[] activationCosts = new double[4];

        ActivationFunction[] activationFunctions= new ActivationFunction[]{
                ActivationFunction.SIGMOID,
                ActivationFunction.TANH,
                ActivationFunction.LEAKYRELU,
                ActivationFunction.LINEAR
        };

        // Vier mal "Antrainieren" mit den verschiedenen Aktivierungsfunktionen
        saveWeightsAndBiases("weights_biases.txt");
        for(int n = 0; n<4; n++) {
            for(NeuronLayer l : hiddenLayers) {
                l.setActivationFunction(activationFunctions[n]);
            }
            loadWeightsAndBiases("weights_biases.txt");
            // "Antrainieren"
            for(int i = 0; i<500; i++) {
                oldCost = calculateCurrentCostSum(trainingData);

                // Altes Netz sichern
                saveWeightsAndBiases("new_weights_biases.txt");

                // Layers anpassen
                for(NeuronLayer l : hiddenLayers) {
                    l.changeValue(oldCost, 0.01); // Hiddenlayer wird geändert
                }

                outputLayer.changeValue(oldCost, 0.01); // Outputlayer wird geändert

                // Neue Kosten berechnen
                newCost = calculateCurrentCostSum(trainingData);

                if (oldCost < newCost) { // Wenn Verschlechterung
                    loadWeightsAndBiases("new_weights_biases.txt");
                    //costs.add(oldCost);
                } else { // Wenn Verbesserung
                    //saveWeightsAndBiases("new_weights_biases.txt");
                    //costs.add(newCost);
                }
            }
            activationCosts[n] = newCost;
        }
        int lowestIndex = 0;
        for(int i = 0; i<4; i++) {
            if(activationCosts[i] < activationCosts[lowestIndex]) {
                lowestIndex = i;
            }
        }
        for(NeuronLayer l : hiddenLayers) {
            l.setActivationFunction(activationFunctions[lowestIndex]); // Hiddenlayers werden auf beste Funktion gesetzt
        }

        System.out.println(activationFunctions[lowestIndex].toString());
    }



    public void training(int trainingSize) {
        double oldCost = 0;
        double newCost = 0;

        try (BufferedReader reader = new BufferedReader(new FileReader("trainingdoc.txt"))) {
            String line;
            line = reader.readLine();
            oldCost = Double.parseDouble(line);
            line = reader.readLine();
            ActivationFunction activationFunction = ActivationFunction.getActivation(line);
            if(oldCost>2) {
                System.out.println("Pretraining wird durchgeführt, da die Kosten >2 sind.");
                preTraining();
            } else {
                System.out.println("Pretraining wird uebersprungen, da die Kosten <=2 sind.");
                for (NeuronLayer hiddenLayer : hiddenLayers) {
                    hiddenLayer.setActivationFunction(activationFunction);
                }
            }
        } catch (IOException e) {
            System.out.println("Pretraining wird durchgeführt, da dies der erste Durchlauf ist.");
            preTraining();
        }

        loadWeightsAndBiases("weights_biases.txt");
        //hiddenLayer.setActivationFunction(ActivationFunction.LINEAR);

        for(int i = 0; i < trainingSize; i++) {
            // Alte Kosten berechnen
            oldCost = calculateCurrentCostSum(trainingData);

            // Altes Netz sichern
            saveWeightsAndBiases("weights_biases.txt");

            // Layer anpassen
            for (NeuronLayer hiddenLayer : hiddenLayers) {
                hiddenLayer.changeValue(oldCost, 0.001); // Hiddenlayer wird geändert
            }
            outputLayer.changeValue(oldCost, 0.001); // Outputlayer wird geändert

            // Neue Kosten berechnen
            newCost = calculateCurrentCostSum(trainingData);

            if (oldCost < newCost) { // Wenn Verschlechterung
                loadWeightsAndBiases("weights_biases.txt");
                costs.add(oldCost);
            } else { // Wenn Verbesserung
                saveWeightsAndBiases("weights_biases.txt");
                costs.add(newCost);
            }
            

        }

        printCost();
        gui.init();
        try (FileWriter file = new FileWriter("trainingdoc.txt")) {
            file.write(Neuron.roundDouble(newCost, 4) + "\n");
            file.write(hiddenLayers.get(0).getActivationFunction().toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Das Training wurde abgeschlossen. Neue Kosten: " + Neuron.roundDouble(costs.getLast(), 4));

    }
/*
    private void changeActivationFunktion(NeuronLayer hidden, NeuronLayer output, List<double[]> trainingData) {
        //Kosten mit allen Möglichkeiten der Aktivierungsfunktionen ausrechen und die Kosten in Array speichern
        Double[] costs = new Double[16];
        ActivationFunction[] activationFunctions= new ActivationFunction[]{
                ActivationFunction.SIGMOID,
                ActivationFunction.TANH,
                ActivationFunction.LEAKYRELU,
                ActivationFunction.LINEAR
        };
        int stelle=0;
        for(int i=0; i<=3; i++) {
            hiddenLayer.setActivationFunction(activationFunctions[i]);
            for(int y = 0; y<=3; y++){
                outputLayer.setActivationFunction(activationFunctions[y]);
                costs[stelle]=calculateCurrentCostSum(trainingData);
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
            hiddenLayer.setActivationFunction(activationFunctions[3]);
            outputLayer.setActivationFunction(activationFunctions[3]);
        } else {
            int hiddenStelle = minIndex / 4 +1;
            int outputStelle = minIndex - ((hiddenStelle-1)*4);
            hiddenLayer.setActivationFunction(activationFunctions[hiddenStelle-1]);
            outputLayer.setActivationFunction(activationFunctions[outputStelle]);
        }
        
    }
    */

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
            for (NeuronLayer hiddenLayer : hiddenLayers) {
                for (Neuron neuron : hiddenLayer.getNeurons()) {
                    file.write(roundDouble(neuron.getBias(), 4) + "\n");
                    file.write(Arrays.toString(neuron.getWeights()) + "\n");
                }
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
            for (NeuronLayer hiddenLayer : hiddenLayers) {
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
            System.err.println("Load Weights and Bias: Datei konnte nich gefunden werden");
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

    /*
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

     */

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

    
    public List<NeuronLayer> getHiddenLayers() {
        return hiddenLayers;
    }
    
    public NeuronLayer getOutputLayer() {
        return outputLayer;
    }
}
