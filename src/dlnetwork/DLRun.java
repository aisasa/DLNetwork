package dlnetwork;

import java.io.*;                   // IOException

public class DLRun {
    // Defining, building and starting the neural network
    public static void main(String[] args) throws IOException, ClassNotFoundException{
        // 1. Loading MNIST data (MNISTStore arrays)
        MNISTStore.loadMNISTArrays();
        
        // 2.a Network constructor parameters
        // Defining neural net shape
        int[] initShape = {784, 200, 10}; 
        // Cost function
        DLNetwork.CostFn costF = DLNetwork.CostFn.CROSS_ENTROPY;    // QUADRATIC | CROSS_ENTROPY   
        // Init learning rate (fine tune of weights/biases variations) and possible adaptation
        double lRate = 0.12;    
        DLNetwork.AdaptLRate adapLR = DLNetwork.AdaptLRate.QUAD;    // NO | LIN | QUAD | SQRT
        // Regularization type and possible parameters
        DLNetwork.Reglz reg = DLNetwork.Reglz.L2;                   // NO | L2   
        double lambda = 1.75;                                       // L2 regularization parameter
        // How to initialize weights and biases sets
        DLInit.WBInitType wbInitType = DLInit.WBInitType.RANDOM;    // RANDOM | RAND_AND_SAVE | LOAD_PRE_SAVED | LOAD_BY_NAME
        // Mini batch: subset of the training set. Here, the lower the better
        int miniBatch = 1;      
        // 2.b Build neural network and show network parameters as reference
        // (int[] shape, double lr, CostFn cFn, Reglz reg, double lmbd, int mB, WBInitType init/load)
        DLNetwork net = new DLNetwork(initShape, lRate, costF, reg, lambda, adapLR, miniBatch, wbInitType); 
        net.paramRef();
        
        // 3.  Execution parameters definition and start computing
        int epochs = 100;           // How many times we treat the entire training data set
        boolean shuffle = true;     // Shuffle or not the training set between epochs
        int minScoreToRef = 9840;   // Minimum score from which to record best results (not mandatory)
        // net.start(epochs, shuffle);              // Do not save models
        net.start(epochs, shuffle, minScoreToRef);  // Save best models from succes = minScoreToRef
    }

}
