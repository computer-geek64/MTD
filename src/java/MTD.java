/**
 * MTD.java
 * Ashish D'Souza
 * @computer-geek64
 * December 20th, 2018
 */

import javax.swing.*;

public class MTD {
    /**
     * @param args the command line arguments
     */
    static Home home;
    static Evaluate evaluate;
    static EvaluateOutput evaluateOutput;
    public static void main(String[] args) {
        // TODO code application logic here
        home = new Home();
        evaluate = new Evaluate();
        evaluateOutput = new EvaluateOutput();
        
        home.setTitle("MTD | Home");
        evaluate.setTitle("MTD | Evaluate");
        evaluateOutput.setTitle("MTD | Evaluate Output");
        
        home.setLocationRelativeTo(null);
        evaluate.setLocationRelativeTo(null);
        evaluateOutput.setLocationRelativeTo(null);
        
        home.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        evaluate.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        evaluateOutput.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        
        home.submit.setEnabled(false);
        home.reset.setEnabled(false);
        evaluate.submit.setEnabled(false);
        evaluate.reset.setEnabled(false);
        
        home.setVisible(true);
        evaluate.setVisible(false);
        evaluateOutput.setVisible(false);
    }
}
