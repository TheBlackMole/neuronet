public class Main {


    public static void main(String[] args) {
        Network network = new Network(1, 1);
        double[] input = {1,2};
        double[] results = network.run(input);
        System.out.println("Ausgabe:");
        for(double result : results) {
            System.out.println(result);
        }
    }

}
