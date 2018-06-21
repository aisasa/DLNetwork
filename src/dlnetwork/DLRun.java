package dlnetwork;

import java.io.*;                   // IOException

public class DLRun {
    
    public static void main(String[] args) throws IOException, ClassNotFoundException{
        
        // Defining neural net shape
        int[] initShape = {784, 100, 10}; 
        // Cost function
        DLNetwork.CostFn costF = DLNetwork.CostFn.CROSS_ENTROPY;    
        // Init learning rate (fine tune of weights/biases variations) and possible adaptation
        double lRate = 0.074;       
        DLNetwork.AdaptLRate adapLR = DLNetwork.AdaptLRate.LIN; //LIN;
        // Regularization type and possible parameters. No reg -> Reglz.NO
        DLNetwork.Reglz reg = DLNetwork.Reglz.L2;   
        double lambda = 1.0;        // L2 regularization parameter
        // How to initialize weights and biases sets
        DLInit.InitType initType = DLInit.InitType.LOAD_PRE_SAVED;
        // Working parameters
        int epochs = 100;           // How many times we treat the entire training data set
        int miniBatch = 1;          // Subset of the training set. Here, the lower the better
        
        // Loading MNIST data in its arrays
        MNISTStore.loadMNISTArrays();
        System.out.println("MNIST data set loaded");
        
        // Creating neural network
        // DLNetwork(int[] shape, double lr, CostFn cFn, Reglz reg, double lmbd, InitType init/load)
        DLNetwork net = new DLNetwork(initShape, lRate, costF, reg, lambda, adapLR, miniBatch, initType); 
        
        // Network parameters as reference
        net.paramRef();
        
        // Start computing
        net.start(epochs);
    }

}
