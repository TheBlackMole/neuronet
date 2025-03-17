import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Backprop {
    private Network network;
    private List<double[]> z;
    private List<double[]> a;
    private List<double[]> biases;
    private List<double[][]> weights;
    private List<ActivationFunction> activationFunctions;
    private List<NeuronLayer> layers;
    private int N; // Anzahl Layer (0,1,2,...,N), entspricht Index der Weights/Bias-Liste

    public Backprop(Network network) {
        // Layers speichern
        this.network = network;
        layers = network.getLayers();
        N = layers.size();

        initBiases();
        initWeights();
        initActivationFunctions();
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

    private void initActivationFunctions() {
        activationFunctions = new ArrayList<>();
        activationFunctions.add(ActivationFunction.SIGMOID);
        for(NeuronLayer l : layers) {
            activationFunctions.add(l.getActivationFunction());
        }
    }

    public double[] calculateZandA(double[] input) {
        z = new ArrayList<>();
        z.add(new double[input.length]);
        a = new ArrayList<>();
        a.add(input);

        for(int k = 1; k<=N; k++) { // Für jedes Hiddenlayer und das Outputlayer
            double[] biasesCurrentLayer = biases.get(k);
            double[][] weightsCurrentLayer = weights.get(k);
            ActivationFunction activationFunction = activationFunctions.get(k);
            double[] zCurrentLayer = new double[biasesCurrentLayer.length];
            double[] aCurrentLayer = new double[zCurrentLayer.length];
            for(int i = 0; i<zCurrentLayer.length; i++) {
                double currentZ = biasesCurrentLayer[i];
                for(int j = 0; j < input.length; j++) {
                    currentZ += weightsCurrentLayer[i][j] * input [j];
                }
                zCurrentLayer[i] = currentZ;
                switch (activationFunction) {
                    case TANH:
                        aCurrentLayer[i] = tanh(currentZ);
                        break;
                    case LEAKYRELU:
                        aCurrentLayer[i] = leakyReLu(currentZ);
                        break;
                    case LINEAR:
                        aCurrentLayer[i] = linear(currentZ);
                        break;
                    default:
                        aCurrentLayer[i] = sigmoid(currentZ);
                        break;
                }
            }
            z.add(zCurrentLayer);
            a.add(aCurrentLayer);
            input = aCurrentLayer;
        }
        return a.get(a.size()-1);
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

    private double tanh(double x) {
        return (2 / (1 + Math.exp(-2 * x))) - 1;
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private double leakyReLu(double x) {
        double a = 0.01; // --> könnte auch flexibel gemacht werden
        return (x >= 0) ? x : a * x;
    }

    private double linear(double x) {
        return x;
    }
}
