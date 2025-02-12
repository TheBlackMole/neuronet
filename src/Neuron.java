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
            weights[i] = rand.nextDouble() * 2 - 1; // Werte zwischen 0.01 und -0.01
            weights[i] = roundDouble(weights[i], 4);
        }
        bias = rand.nextDouble() * 2 - 1;
        bias = roundDouble(bias, 4);
        /*System.out.println("Neues Neuron erstellt mit Bias: " + bias + ", Weights:");
        for(double weight : weights) {
            System.out.println(weight);
        } */
    }

    public double activate(String type, double ratio, double[] inputs) {
        //System.out.println("Inputs: " + inputs.length + " weights: "+ weights.length);
        if (inputs.length != weights.length) {
            throw new IllegalArgumentException("Eingangsgröße stimmt nicht mit Gewichtsanzahl überein.");
        }
        double sum = bias;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }
        sum = roundDouble(sum, 5);

        if (type.equalsIgnoreCase("tanh")) { // -1; 1
            sum = tanh(sum * ratio);
        } else if (type.equalsIgnoreCase("sigmoid")) { // 0; 1
            sum = sigmoid(sum * ratio);
        } else if (type.equalsIgnoreCase("linear")) {
            sum = linear(sum *ratio);
        } else if (type.equalsIgnoreCase("leakyReLu")) { //  { a*x ; x } --> oft x=0.01 --> für negative x Werte
            sum = leakyReLu(sum * ratio);
        } else {
            System.out.println("fehlerhafter Aufruf der Aktivierungsfunktion");
            return 0;
        }
        /*
        if (typ == "softmax") { // Wahrscheinlichkeit --> oft für Output --> Summe immer 1
            sum = softmax(sum * ratio);
        }
        */ // --> muss in arrays zurückgegeben werden --> also seperat, weil abhängig von den anderen Neuronen

        /*
         * INFOS - WANN WElCHLE FUNKTION
         * - Für versteckte Schichten wird oft ReLU verwendet. Diese Funktion ist einfach und beschleunigt das Training.
         * - Bei der Klassifikation von zwei Klassen ist Sigmoid die beste Wahl. Es gibt eine klare Entscheidung zwischen den Klassen.
         * - Bei mehreren Klassen ist Softmax in der Ausgabeschicht am besten. Es bestimmt die Klassenzugehörigkeit für mehrere Kategorien.
         */
        return sum;
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

    private double leakyReLu(double x) {
        double a = 0.01; // --> könnte auch flexibel gemacht werden
        return (x >= 0) ? x : a * x;
    }

    private double linear(double x) {
        return x;
    }

    public double getBias() {
        return roundDouble(bias, 4);
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