import java.util.Random;

public class Neuron {
    private double[] weights;
    private double bias;

    public Neuron(int inputSize) {
        Random rand = new Random();
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            weights[i] = rand.nextDouble() * 2 - 1; // Werte zwischen -1 und 1
        }
        bias = rand.nextDouble() * 2 - 1;
    }

    public double activate(double[] inputs) {
        if (inputs.length != weights.length) {
            throw new IllegalArgumentException("Eingangsgröße stimmt nicht mit Gewichtsanzahl überein.");
        }

        double sum = bias;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }

        return sigmoid(sum);
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
}