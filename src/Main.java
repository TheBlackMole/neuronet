import java.util.*;

public class Main {



    public static void main(String[] args) {
        // Hier kann man die Größe der Hiddenlayers angebe, {4,4} -> 2 Hiddenlayer mit je 4 Neuronen
        Integer[] h = new Integer[] {3,3};
        // Umwandlung in Arraylist
        List<Integer> hiddenLayerSizes = new ArrayList<>();
        Collections.addAll(hiddenLayerSizes, h);

        // Erstellen und Trainieren des Network
        Network network = new Network(2, hiddenLayerSizes, 1, 1000);
        network.training(1000);

        // Ausgabe
        double[] input = {10, 10};
        double[] results = network.run(input);
        System.out.println("Input " + Arrays.toString(input));
        for(double result : results) {
            System.out.println(Neuron.roundDouble(result,3));
        }
        System.out.println("\n\n\n");
        Random r = new Random();
        for(int i = 0; i < 50; i++) { // Verschieden In- und Outputs testen
            int x = r.nextInt(100) + 1;
            //int y = r.nextInt(100) + 1;
            int y = x - i;
            input[0] = x;
            input[1] = y;
            results = network.run(input);
            System.out.println("Input " + Arrays.toString(input));
            System.out.println(Neuron.roundDouble(results[0],3));
        }


    }

}
