import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Random;

public class Neuron {
    private double[] weights;
    private double bias;

    public Neuron(int inputSize) {
        Random rand = new Random();
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            weights[i] = rand.nextDouble() * 0.02 - 0.01; // Werte zwischen 0.01 und -0.01
        }
        bias = rand.nextDouble() * 0.02 - 0.01;
        /*System.out.println("Neues Neuron erstellt mit Bias: " + bias + ", Weights:");
        for(double weight : weights) {
            System.out.println(weight);
        } */
    }

    public double activate(double[] inputs) {
        //System.out.println("Inputs: " + inputs.length + " weights: "+ weights.length);
        if (inputs.length != weights.length) {
            throw new IllegalArgumentException("Eingangsgröße stimmt nicht mit Gewichtsanzahl überein.");
        }

        double sum = bias;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }
        sum = roundDouble(sum, 5);
        double smallSum = sum / 4;
        /*
        System.out.println("sum: " + sum);
        System.out.println("sigmoid small: " + roundDouble(sigmoid(smallSum),4));
        System.out.println("tanh small: " + roundDouble(tanh(smallSum),4));
        System.out.println("sigmoid: " + roundDouble(sigmoid(sum),4));
        System.out.println("tanh: " + roundDouble(tanh(sum),4));
        */
        return tanh(sum);
        //return sigmoid(smallSum);
    }
    private double roundDouble(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();

        BigDecimal bd = new BigDecimal(Double.toString(value));
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }

    private double tanh(double x) {
        return (2 / (1 + Math.exp(-2 * x))) - 1;
    } 

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double[] getWeights() {
        return weights;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }
}