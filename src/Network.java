import java.util.ArrayList;
import java.util.List;

public class Network {

    private List<Neuron> hiddenLayer;
    private List<Neuron> outputLayer;

    public Network(int hiddenLayerCount, int outputLayerCount) {
        hiddenLayer = new ArrayList<>();
        outputLayer = new ArrayList<>();
        for(int i = 0; i < hiddenLayerCount; i++) {
            hiddenLayer.add(new Neuron(2));
        }
        for(int i = 0; i < outputLayerCount; i++) {
            outputLayer.add(new Neuron(hiddenLayerCount));
        }
    }

    public double[] run(double[] input) {
        System.out.println("Inputs: " + input [0] + " " + input[1]);

        double[] resultsHiddenLayer = new double[hiddenLayer.size()];
        int i = 0;
        for(Neuron n : hiddenLayer) {
            resultsHiddenLayer[i] = n.activate(input);
            i++;
        }

        i = 0;
        double[] resultsOutputLayer = new double[outputLayer.size()];
        for(Neuron n : outputLayer) {
            resultsOutputLayer[i] = n.activate(resultsHiddenLayer);
            i++;
        }

        return resultsOutputLayer;
    }
}
