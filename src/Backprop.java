import java.util.ArrayList;
import java.util.List;

public class Backprop {
    private Network network;
    private List<double[]> neurons;
    private List<double[]> biases;
    private List<double[][]> weights;
    private List<NeuronLayer> hiddenLayers;
    private NeuronLayer outputLayer;

    public Backprop(Network network) {
        this.network = network;
        // Layers speichern
        hiddenLayers = network.getHiddenLayers();
        outputLayer = network.getOutputLayer();

        initBiases();
        initWeights();
    }

    private void initBiases() {
        // Biase in Matrizen speichern
        biases = new ArrayList<>();
        biases.add(new double[network.getInputLayerSize()]); // Leeres Array, da das Inputlayer keine Biases hat
        for(NeuronLayer l : hiddenLayers) {
            Neuron[] hiddenLayerNeurons = l.getNeurons();
            double[] hiddenLayerBiases = new double[hiddenLayerNeurons.length];
            for(int i = 0; i < hiddenLayerNeurons.length; i++) { // Biase aus Neuronenarray uebernehmen
                hiddenLayerBiases[i] = hiddenLayerNeurons[i].getBias();
            }
            biases.add(hiddenLayerBiases);
        }
        Neuron[] outputLayerNeurons = outputLayer.getNeurons();
        double[] outputLayerBiases = new double[outputLayerNeurons.length];
        for(int i = 0; i < outputLayerNeurons.length; i++) { // Biase aus Neuronenarray uebernehmen
            outputLayerBiases[i] = outputLayerNeurons[i].getBias();
        }
        biases.add(outputLayerBiases);
    }

    private void initWeights() {
        // Weights in Matrizen speichern
        weights = new ArrayList<>();
        weights.add(new double[network.getInputLayerSize()][1]); // Leeres Array, da das Inputlayer keine Weights hat
        for(int n = 0; n < hiddenLayers.size(); n++) {
            Neuron[] hiddenLayerNeurons = hiddenLayers.get(n).getNeurons();
            int layerSizeBefore = 0;
            if(n == 0) { // beim ersten HL muss das IL als Vorgaenger genommen werden
                layerSizeBefore = network.getInputLayerSize();
            } else {
                layerSizeBefore = hiddenLayers.get(n-1).getNeurons().length;
            }
            double[][] hiddenLayerWeights = new double[hiddenLayerNeurons.length][layerSizeBefore];
            for(int i = 0; i < hiddenLayerNeurons.length; i++) { // Biase aus Neuronenarray uebernehmen
                for(int j = 0; j < layerSizeBefore; j++) {
                    hiddenLayerWeights[i][j] = hiddenLayerNeurons[i].getWeights()[j];
                }
            }
            weights.add(hiddenLayerWeights);
        }
        Neuron[] outputLayerNeurons = outputLayer.getNeurons();
        int lastHiddenLayerSize = hiddenLayers.get(hiddenLayers.size() - 1).getNeurons().length;
        double[][] outputLayerWeights = new double[outputLayerNeurons.length][lastHiddenLayerSize];
        for(int i = 0; i < outputLayerNeurons.length; i++) { // Biase aus Neuronenarray uebernehmen
            for(int j = 0; j < lastHiddenLayerSize; j++) {
                outputLayerWeights[i][j] = outputLayerNeurons[i].getWeights()[j];
            }
        }
        weights.add(outputLayerWeights);
    }

    public double getBias(int layer, int indexNeuron) {
        return biases.get(layer)[indexNeuron];
    }

    public void setBias(int layer, int indexNeuron, double value) {
        biases.get(layer)[indexNeuron] = value;
    }

    public double getWeight(int layer, int indexNeuron, int indexPrevNeuron) {
        return weights.get(layer)[indexNeuron][indexPrevNeuron];
    }

    public void getWeight(int layer, int indexNeuron, int indexPrevNeuron, double value) {
        weights.get(layer)[indexNeuron][indexPrevNeuron] = value;
    }
}
