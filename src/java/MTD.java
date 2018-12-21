/** MTD.java
 * Ashish D'Souza (computer-geek64)
 * December 20th, 2018
 */
import java.util.Date;
import javax.swing.*;

public class MTD {
    /**
     * @param args the command line arguments
     */
    static Home home;
    static Evaluate evaluate;
    public static void main(String[] args) {
        // TODO code application logic here
        home = new Home();
        evaluate = new Evaluate();
        
        home.setTitle("MTD | Home");
        evaluate.setTitle("MTD | Evaluate");
        
        home.setLocationRelativeTo(null);
        evaluate.setLocationRelativeTo(null);
        
        home.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        evaluate.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        
        home.submit.setEnabled(false);
        home.reset.setEnabled(false);
        
        evaluate.formattedTextField.setValue(new Date(2018 - 1900, 12, 20));
        
        home.setVisible(true);
        evaluate.setVisible(false);
    }
}
