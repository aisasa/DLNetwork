package dlnetwork;

import java.io.*;                   // IOException

public class DLRun {
    // Defining, building and starting the neural network
    public static void main(String[] args) throws IOException, ClassNotFoundException{
        // Network constructor parameters
        // Defining neural net shape
        int[] initShape = {784, 100, 10}; 
        // Cost function
        DLNetwork.CostFn costF = DLNetwork.CostFn.CROSS_ENTROPY; //QUADRATIC;    
        // Init learning rate (fine tune of weights/biases variations) and possible adaptation
        double lRate = 0.074;       
        DLNetwork.AdaptLRate adapLR = DLNetwork.AdaptLRate.LIN; //LIN;
        // Regularization type and possible parameters. No reg -> Reglz.NO
        DLNetwork.Reglz reg = DLNetwork.Reglz.L2;   
        double lambda = 1.0;        // L2 regularization parameter
        // How to initialize weights and biases sets
        DLInit.WBInitType initType = DLInit.WBInitType.RAND_AND_SAVE; // RANDOM, RAND_AND_SAVE, LOAD_PRE_SAVED, LOAD_BEST
        // Mini batch: subset of the training set. Here, the lower the better
        int miniBatch = 1;      
        // Start/working computing parameters
        int epochs = 100;           // How many times we treat the entire training data set
        boolean shuffle = false;    // Shuffle or not training sets between epochs
        
        // Loading MNIST data in its arrays
        MNISTStore.loadMNISTArrays();
        System.out.println("MNIST data set loaded");
        
        // Create neural network and show network parameters as reference
        // (int[] shape, double lr, CostFn cFn, Reglz reg, double lmbd, int mB, WBInitType init/load)
        DLNetwork net = new DLNetwork(initShape, lRate, costF, reg, lambda, adapLR, miniBatch, initType); 
        net.paramRef();
        
        // Start computing
        net.start(epochs, shuffle);
    }

}
