//import java.io.FileWriter;
//import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.List;
import java.util.Random;



public class Network {

    private NeuronLayer hiddenLayer;
    private NeuronLayer outputLayer;

    public Network(int inputLayerSize, int hiddenLayerSize, int outputLayerSize) {
        hiddenLayer = new NeuronLayer(hiddenLayerSize, inputLayerSize);
        outputLayer = new NeuronLayer(outputLayerSize, hiddenLayerSize);
        loadWeightsAndBiases("weights_biases.txt");
    }

    public double[] run(double[] input) {
        System.out.println("Inputs: " + input [0] + " " + input[1]);

        double[] resultsHiddenLayer = hiddenLayer.forward(input);
        return outputLayer.forward(resultsHiddenLayer);
    }

    public void training(int trainingSize, List<double[]> trainingData) {
        Random r = new Random();
        double BiasOutStart = outputLayer.getNeuron(0).getBias();
        double BiasHiddenStart = hiddenLayer.getNeuron(0).getBias();
        

        for(int i = 0; i < trainingSize; i++) {
            System.out.println("Bias out start: " + roundDouble(BiasOutStart, 3));
            System.out.println("Bias hidden start: " + roundDouble(BiasHiddenStart, 3));
            // Testdaten mit altem Netz durchführen und Kosten berechnen
            //System.out.println("HL. n. : " + hiddenLayer.getNeuron(0));
            double[] currentData = trainingData.get(r.nextInt(trainingData.size()));
            double[] input = {currentData[0], currentData[1]};
            double[] results = run(input);
            //double oldResult = results[0];
            //System.out.println("result: " + results[0]); 
            double[] trainingDataResults = {currentData[2]};
            //System.out.println("trainingResult: " + trainingDataResults[0]);
            double oldCost = calculateCostFunction(results, trainingDataResults);

            // Altes Netz sichern
            NeuronLayer oldHiddenLayer = hiddenLayer.getCopy();
            NeuronLayer oldOutputLayer = outputLayer.getCopy();

            // Schleife zur Verbesserung
            boolean goOn = true;
            int barrier = 0; // Zählvariable, bricht nach 10 mal ab
            double newCost = 0; // Kosten nach Änderung des Outputlayers
            double veryNewCost = 0; // Kosten nach Änderung des Hiddenlayers
            while (goOn == true) {
                barrier++;
                //System.out.println("barrier: " + barrier);
                outputLayer.changeValue(oldCost); // Outputlayer wird geändert und neue Kosten berechnet
                results = run(input);
                newCost = calculateCostFunction(results, trainingDataResults);
                //double newResult = results[0];
                //double oldDelta = trainingDataResults[0] - oldResult;
                //double newDelta = trainingDataResults[0] - newResult;
                //System.out.println("oldresult: " + oldResult + " newresult: " + newResult);
                //System.out.println("oldDelta: " + oldDelta + " newDelta: " + newDelta);
                //System.out.println("trainingResult: " + trainingDataResults[0])
                double hiddenBias = hiddenLayer.getNeuron(0).getBias();
                System.out.println("hiddenLayer o. Neuron Bias: " + roundDouble(hiddenBias,3));
                System.out.println("outputLayer o. Neuron Bias: " + roundDouble(outputLayer.getNeuron(0).getBias() , 3) );;
                if (oldCost > newCost && newCost != 0) { // Wenn Verbesserung
                    hiddenLayer.changeValue(newCost); // Hiddenlayer wird geändert und neue Kosten berechnet
                    results = run(input);
                    veryNewCost = calculateCostFunction(results, trainingDataResults);
                    if(newCost < veryNewCost) {
                        hiddenLayer = oldHiddenLayer; // Schlechter geworden -> Rückgängig machen
                    }
                } else {
                    outputLayer = oldOutputLayer; // Schlechter geworden -> Rückgängig machen
                }
                
                if(barrier>=10 || veryNewCost == 0){
                    goOn = false;
                    break;
                }
            }
            
            
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
        }
        saveWeightsAndBiases("weights_biases.txt");
        System.out.println("Neoron Out.0 Bias ende: " + roundDouble(outputLayer.getNeuron(0).getBias(), 3));
        System.out.println("Neuron hidden.0 Bias ende: " + roundDouble(hiddenLayer.getNeuron(0).getBias(), 3));
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
}
