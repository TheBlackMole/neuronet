public enum ActivationFunction {
    TANH,
    SIGMOID,
    LINEAR,
    LEAKYRELU;

    public static ActivationFunction getActivation(String activation) {
        if(activation.equalsIgnoreCase("TANH")) return TANH;
        if(activation.equalsIgnoreCase("SIGMOID")) return SIGMOID;
        if(activation.equalsIgnoreCase("LINEAR")) return LINEAR;
        if(activation.equalsIgnoreCase("LEAKYRELU")) return LEAKYRELU;
        return null;
    }
}




