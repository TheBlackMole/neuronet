//import java.util.Arrays;
import java.util.Random;

public class NeuronLayer {
    private Neuron[] neurons;
    private ActivationFunction activationFunction;

    public NeuronLayer(int numNeurons, int inputSize) {
        neurons = new Neuron[numNeurons];
        for (int i = 0; i < numNeurons; i++) {
            neurons[i] = new Neuron(inputSize);
        }
        activationFunction = ActivationFunction.SIGMOID;
    }

    public double[] forward(double[] input, ActivationFunction type) {
        ActivationFunction activationF;
        if (type != null) {
            activationF = type;
        } else{
            activationF = activationFunction;
        }
        double[] output = new double[neurons.length];
        int i = 0;
        for(Neuron n : neurons) {
            output[i] = n.activate(activationF, 0.25 , input);
            i++;
        }
        return output;

        /* Alte Methode, kp wie das funktioniert
        return Arrays.stream(neurons)
                .mapToDouble(neuron -> neuron.activate(inputs))
                .toArray();
        */
    }
    public void changeValue(double cost) { // Ändert das Bias und EIN Weight eines zufälligen Neurons aus dem Layer
        if (cost == 0) { // Wichtig: Abhängig von den Kosten, + oder - ist zufällig
            return;
        }
        Random r = new Random();
        double change;
        if (r.nextBoolean()) {
            change = 0.01;
        } else {
            change = -0.01;
        }
        int index = r.nextInt(neurons.length);
        // Bias verändern
        neurons[index].setBias(neurons[neurons.length-1].getBias() + change * cost);
        // Zufälliges Weigth verändern
        double[] weights = neurons[index].getWeights();
        int indexWeights = r.nextInt(weights.length);
        weights[indexWeights] += change * cost;
        neurons[index].setWeights(weights);
    }
    
    /*
    public void changeSingleValue(double cost) { // Aktuell nicht in Benutzung
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
     */

    public void setNeurons(Neuron[] neurons) {
        this.neurons = neurons;
    }

    public Neuron[] getNeurons() {
        return neurons;
    }
    public Neuron getNeuron(int i) {
        return neurons[i];
    }
    public void setActivationFunction(ActivationFunction a) {
        this.activationFunction=a;
    }
    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public NeuronLayer getCopy() {
        /* NeuronLayer neuronLayer = new NeuronLayer(neurons.length, neurons[0].getWeights().length);
        neuronLayer.setNeurons(neurons);
        return neuronLayer;
        */
        return this;
    }
}
