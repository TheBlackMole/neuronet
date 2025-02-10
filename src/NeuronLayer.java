import java.util.Arrays;
import java.util.Random;

public class NeuronLayer {
    private Neuron[] neurons;

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
    public void changeValue(double result, double trainingResult) {
        if (result==trainingResult) {
            return;
        }
        if (result > trainingResult) {
            double change = 0.1;
            neurons[neurons.length-1].setBias(neurons[neurons.length-1].getBias() - change);
        }
        if (result < trainingResult) {
            double change = 0.1;
            neurons[neurons.length-1].setBias(neurons[neurons.length-1].getBias() + change);
        }
    }
    

    public void changeSingleValue(double cost) {
        Random r = new Random();
        double change = r.nextDouble() * 0.2 - 0.1; // zufallszahl zw. -0.1 und 0.1
        int index = r.nextInt(neurons.length);
        boolean type = r.nextBoolean();
        if(type) { // Type = Bias
            neurons[index].setBias(neurons[index].getBias() + change * cost);
        } else { // Type = Weight
            double[] weights = neurons[index].getWeights();
            int indexWeights = r.nextInt(weights.length);
            weights[indexWeights] += change * cost;
            neurons[index].setWeights(weights);
        }
    }

    public void setNeurons(Neuron[] neurons) {
        this.neurons = neurons;
    }

    public Neuron[] getNeurons() {
        return neurons;
    }
    public Neuron getNeuron(int i) {
        return neurons[i];
    }

    public NeuronLayer getCopy() {
        /* NeuronLayer neuronLayer = new NeuronLayer(neurons.length, neurons[0].getWeights().length);
        neuronLayer.setNeurons(neurons);
        return neuronLayer;
        */
        return this;
    }
}
