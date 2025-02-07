public class Network {

    private final NeuronLayer hiddenLayer;
    private final NeuronLayer outputLayer;

    public Network(int inputLayerSize, int hiddenLayerSize, int outputLayerSize) {
        hiddenLayer = new NeuronLayer(hiddenLayerSize, inputLayerSize);
        outputLayer = new NeuronLayer(outputLayerSize, hiddenLayerSize);
    }

    public double[] run(double[] input) {
        System.out.println("Inputs: " + input [0] + " " + input[1]);

        double[] resultsHiddenLayer = hiddenLayer.forward(input);
        return outputLayer.forward(resultsHiddenLayer);
    }
}
