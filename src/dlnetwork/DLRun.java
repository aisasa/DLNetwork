package dlnetwork;

import java.io.*;                   // Write and read files
import java.util.*;                 // ArrayList and Arrays

public class DLRun {
    
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        
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
        DLNetwork net = new DLNetwork(initShape, lRate, costF, reg, lambda, adapLR, initType); 
        
        // Network parameters reference
        System.out.println("Network reference: ");
        System.out.println("· Shape: " + Arrays.toString(net.netShape));
        System.out.println("· Cost function: " + net.costFunction);
        System.out.println("· Learning rate: " + net.learnRate);
        System.out.println("· Adaptative learning rate: " + net.adaptLearnRate);
        if(net.adaptLearnRate == DLNetwork.AdaptLRate.LIN){
            System.out.println("    Error threshold: " + DLNetwork.ERR_THR);
            System.out.println("    Minimum learn rate: " + DLNetwork.MIN_LRN_R);
        }
        System.out.println("· Regularization: " + net.regularization);
        if(net.regularization == DLNetwork.Reglz.L2)
            System.out.println("    L2 lambda: " + net.lambda);
        System.out.println("· Epochs: " + epochs);
        System.out.println("· Mini batch: " + miniBatch);
        System.out.println("· Initialice weights/biases: " + net.initWBType);
        
        // Start computing
        System.out.println("Starting computation: " + new java.util.Date());
        net.start(epochs, miniBatch);
        System.out.println("End computation: " + new java.util.Date());
    }

}
