import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.util.Arrays;
import java.awt.Color;
import java.awt.Font;


public class GUI implements ActionListener   {
    private NeuronLayer hiddenLayer;
    private NeuronLayer outputLayer;
    private Network network;

    private JFrame start;
    private JLabel L_hiddenLayer;
    private JLabel L_outputLayer;
    private JPanel P_input;
    private JTextField TF_input1;
    private JTextField TF_input2;
    private JButton B_los;
    private JLabel L_result;

    public GUI(NeuronLayer hiddenLayer, NeuronLayer outputLayer) {
        this.hiddenLayer=hiddenLayer;
        this.outputLayer=outputLayer;

        start = new JFrame();
        start.setLayout(new GridLayout());
       // start.setLayout(new GridLayout(1, 2));
        start.setSize(900, 600);
        start.setBackground(Color.LIGHT_GRAY);
        start.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        start.setTitle("GUI Start");

        P_input = new JPanel();
        P_input.setLayout(new GridLayout(0,1));
        
        TF_input1 = new JTextField("erster Input");
        TF_input2 = new JTextField("zweiter Input");
        B_los = new JButton("Berechnen");
        
        B_los.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                // TODO Auto-generated method stub
                //throw new UnsupportedOperationException("Unimplemented method 'actionPerformed'");

            
                    String Si1 = TF_input1.getText();
                    String Si2 = TF_input2.getText();
                    double i1 = Double.parseDouble(Si1);
                    double i2 = Double.parseDouble(Si2);

                    double[] inputs = new double[2];
                    inputs[0] = i1;
                    inputs[1] = i2;

                    double[] results = network.run(inputs);
                    String S_results = Arrays.toString(results);
                    L_result.setText(S_results);

                
            }
        });

        P_input.add(TF_input1);
        P_input.add(TF_input2);
        P_input.add(B_los);

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

        L_result = new JLabel("Ergebnis: ");
        L_result.setFont(new Font("helvetica", Font.BOLD, 20));
        L_result.setHorizontalAlignment(JLabel.CENTER);
        L_result.setOpaque(true);

        TF_input1.addFocusListener(new FocusListener() {
            @Override
            public void focusGained(FocusEvent e) {
                TF_input1.setText("");
            }

            @Override
            public void focusLost(FocusEvent e) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("erster Input");
            }
        });

        TF_input2.addFocusListener(new FocusListener() {
            @Override
            public void focusGained(FocusEvent e) {
                TF_input2.setText("");
            }

            @Override
            public void focusLost(FocusEvent e) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("erster Input");
            }
        });

        start.add(P_input);
    
        start.add(neuronWeightsPanel);
        start.add(P_neuronBiasHidden);
        
        start.add(P_neuronWeightsOutput);
        start.add(P_neuronBiasOutput);
        start.add(L_result);
        start.setVisible(true);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'actionPerformed'");
    }

    
}
