import java.util.Arrays;

public class NeuronLayer {
    private final Neuron[] neurons;

    public NeuronLayer(int numNeurons, int inputSize) {
        neurons = new Neuron[numNeurons];
        for (int i = 0; i < numNeurons; i++) {
            neurons[i] = new Neuron(inputSize);
        }
    }

    public double[] forward(double[] inputs) {
        return Arrays.stream(neurons)
                .mapToDouble(neuron -> neuron.activate(inputs))
                .toArray();
    }
}
