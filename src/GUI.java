import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import java.awt.GridLayout;
import java.awt.Color;
import java.awt.Font;


public class GUI extends JFrame  {
    private NeuronLayer hiddenLayer;
    private NeuronLayer outputLayer;

    private JFrame start;
    private JLabel L_hiddenLayer;
    private JLabel L_outputLayer;

    public GUI(NeuronLayer hiddenLayer, NeuronLayer outputLayer) {
        this.hiddenLayer=hiddenLayer;
        this.outputLayer=outputLayer;

        start = new JFrame();
        start.setLayout(new GridLayout());
       // start.setLayout(new GridLayout(1, 2));
        start.setSize(700, 600);
        start.setBackground(Color.LIGHT_GRAY);
        start.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        start.setTitle("GUI Start");

        L_hiddenLayer = new JLabel("HiddenLayer");
        L_hiddenLayer.setFont(new Font("helvetica", Font.BOLD, 20));
        L_hiddenLayer.setHorizontalAlignment(JLabel.CENTER);
        L_hiddenLayer.setOpaque(true);

        L_outputLayer = new JLabel("HiddenLayer");
        L_outputLayer.setFont(new Font("helvetica", Font.BOLD, 20));
        L_outputLayer.setHorizontalAlignment(JLabel.CENTER);
        L_outputLayer.setOpaque(true);

        // Panel für jedes Neuron im hiddenLayer
        JPanel neuronWeightsPanel = new JPanel();
        neuronWeightsPanel.setLayout(new GridLayout(0, 1));
        int stelleNeuron = 0;
        for (Neuron  neuron : hiddenLayer.getNeurons()) {
            stelleNeuron++;
            double[] d_weights = neuron.getWeights();
            for(int i = 0; i < d_weights.length; i++) {
                JLabel L_weightsHidden = new JLabel("Weight " + (stelleNeuron ) + "." + (i+1) + ": " + d_weights[i]);
                neuronWeightsPanel.add(L_weightsHidden);
            }
            
        }

        JPanel P_neuronBiasHidden = new JPanel();
        P_neuronBiasHidden.setLayout(new GridLayout(0,1));
        for (Neuron neuron : hiddenLayer.getNeurons()) {
            JLabel L_bias = new JLabel( "Bias: " + neuron.getBias());
            P_neuronBiasHidden.add(L_bias);
        }

        // Panel für jedes Neuron im outputLayer
        JPanel P_neuronWeightsOutput = new JPanel();
        P_neuronWeightsOutput.setLayout(new GridLayout(0, 1));
        int stelleNeuronOutput = 0;
        for (Neuron  neuron : outputLayer.getNeurons()) {
            stelleNeuronOutput++;
            double[] weights = neuron.getWeights();
            for(int i = 0; i < weights.length; i++) {
                JLabel L_weightsOutput = new JLabel("Weight " + (stelleNeuronOutput ) + "." + (i+1) + ": " + weights[i]);
                P_neuronWeightsOutput.add(L_weightsOutput);
            }
            
        }

        JPanel P_neuronBiasOutput = new JPanel();
        P_neuronBiasOutput.setLayout(new GridLayout(0,1));
        for (Neuron neuron : outputLayer.getNeurons()) {
            JLabel L_bias = new JLabel( "Bias: " + neuron.getBias());
            P_neuronBiasOutput.add(L_bias);
        }

        start.add(neuronWeightsPanel);
        start.add(P_neuronBiasHidden);
        
        start.add(P_neuronWeightsOutput);
        start.add(P_neuronBiasOutput);
        //start.add(neuronPanel);
        
        //start.add(L_outputLayer);
        //start.add(L_hiddenLayer);
        start.setVisible(true);
    }
}
