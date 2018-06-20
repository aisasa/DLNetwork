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
        double lambda = 1.;        // L2 regularization parameter
        // Working parameters
        int epochs = 100;           // How many times we treat the entire training data set
        int miniBatch = 1;          // Subset of the training set. Here, the lower the better
        DLInit.InitType initType = DLInit.InitType.LOAD_PRE_SAVED;
        
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
        
        
        
        
        
        // Defining array x (which will contain each training example input)
        //net.x = new double[MNISTStore.getInputSize()][1];
        
        // For the entire training set, and for several times (epochs)...
        for(int i =0; i<epochs; i++){
            // ...go SGD passing an appropriate size of minibatch...
            net.doSGD(miniBatch);
            // ...shuffling then the set,...
            MNISTStore.shuffleMNIST();
            // ...testing results after each epoch...
            System.out.println("Epoch " + i + ": " + net.doTest()*100/10000 + "%");  
            // ...and go for another one.
        }
    }

}
