import java.util.ArrayList;
import java.util.List;

public class Backprop {
    private Network network;
    private List<double[]> neurons;
    private List<double[]> biases;
    private List<double[][]> weights;
    private List<NeuronLayer> layers;

    public Backprop(Network network) {
        this.network = network;
        // Layers speichern
        layers = network.getLayers();

        initBiases();
        initWeights();
    }

    private void initBiases() {
        // Biase in Matrizen speichern
        biases = new ArrayList<>();
        biases.add(new double[network.getInputLayerSize()]); // Leeres Array, da das Inputlayer keine Biases hat
        for(NeuronLayer l : layers) {
            Neuron[] layerNeurons = l.getNeurons();
            double[] layerBiases = new double[layerNeurons.length];
            for(int i = 0; i < layerNeurons.length; i++) { // Biase aus Neuronenarray uebernehmen
                layerBiases[i] = layerNeurons[i].getBias();
            }
            biases.add(layerBiases);
        }
    }

    private void initWeights() {
        // Weights in Matrizen speichern
        weights = new ArrayList<>();
        weights.add(new double[network.getInputLayerSize()][1]); // Leeres Array, da das Inputlayer keine Weights hat
        for(int n = 0; n < layers.size(); n++) {
            Neuron[] layerNeurons = layers.get(n).getNeurons();
            int layerSizeBefore = 0;
            if(n == 0) { // beim ersten HL muss das IL als Vorgaenger genommen werden
                layerSizeBefore = network.getInputLayerSize();
            } else {
                layerSizeBefore = layers.get(n-1).getNeurons().length;
            }
            double[][] hiddenLayerWeights = new double[layerNeurons.length][layerSizeBefore];
            for(int i = 0; i < layerNeurons.length; i++) { // Weights aus Neuronenarray uebernehmen
                for(int j = 0; j < layerSizeBefore; j++) {
                    hiddenLayerWeights[i][j] = layerNeurons[i].getWeights()[j];
                }
            }
            weights.add(hiddenLayerWeights);
        }
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
