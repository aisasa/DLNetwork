package dlnetwork;

import mnist2JArrays.*;                     // MNIST txt data to Java arrays
import java.io.*;                           // IOException, File, I/O streams
import java.util.zip.*;                     // Zip/unzip files

/**
 * A warehouse to store MNIST Java data structures.
 * 
 * @version 0.9
 * @author  Agustin Isasa Cuartero
 */
public class MNISTStore {
    private static double[][] trainingDataIn;   
    private static double[] trainingDataOut; 
    private static double[][] validationDataIn;
    private static double[] validationDataOut;
    private static double[][] testDataIn;
    private static double[] testDataOut;
    
    /**
     * Transfer MNIST data from previously saved files to proper Java data
     * structures
     * 
     * @throws IOException              If file does not exist
     * @throws ClassNotFoundException   If array does not exist in saved files
     */
    public static void loadMNISTArrays() throws IOException, ClassNotFoundException{
        File fTrainingIn = new File("trainingIn.dat");
        File fTrainingOut = new File("trainingOut.dat");
        File fValidationIn = new File("validationIn.dat");
        File fValidationOut = new File("validationOut.dat");
        File fTestIn = new File("testIn.dat");
        File fTestOut = new File("testOut.dat");
        if(!fTrainingIn.exists() || !fValidationIn.exists() || !fTestIn.exists())
            MNIST2JArrays.mnist2JArrays();
        
        // Training data
        try(ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(new FileInputStream(fTrainingIn)))){ 
            trainingDataIn = (double[][])in.readObject();
        }
        try(ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(new FileInputStream(fTrainingOut)))){ 
            trainingDataOut = (double[])in.readObject();
        }
        // Validation data
        try(ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(new FileInputStream(fValidationIn)))){ 
            validationDataIn = (double[][])in.readObject();
        }
        try(ObjectInputStream in = new ObjectInputStream(
                new GZIPInputStream(new FileInputStream(fValidationOut)))){ 
            validationDataOut = (double[])in.readObject();
        }
        // Test data
        try(ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(new FileInputStream(fTestIn)))){ 
            testDataIn = (double[][])in.readObject();
        }
        try(ObjectInputStream in = new ObjectInputStream(
                new GZIPInputStream(new FileInputStream(fTestOut)))){ 
            testDataOut = (double[])in.readObject();
        }
        System.out.println("MNIST data set loaded");
    }
    
    // Shuffle data sets
    /**
     * A procedure to shuffle data training set to get stochastic improvements 
     * in training execution.
     * 
     */
        public static void shuffleMNIST(){
        DLMath.shuffle(trainingDataIn, trainingDataOut);
    }
    
    // Object getters
    /**
     *
     * @return  A bidimensional array containing the input data from MNIST 
     * training data set.
     */
        public static double[][] getTrainingDataIn(){
        return trainingDataIn;
    }
    /**
     *
     * @return  An array containing the real outputs of each example in MNIST 
     * training data set.
     */
    public static double[] getTrainingDataOut(){
        return trainingDataOut;
    }
    /**
     *
     * @return  A bidimensional array containing the input data from MNIST 
     * validation data set.
     */
    public static double[][] getValidationDataIn(){
        return validationDataIn;
    }
    /**
     *
     * @return  An array containing the real outputs of each example in MNIST 
     * validation data set.
     */
    public static double[] getValidationDataOut(){
        return validationDataOut;
    }
    /**
     *
     * @return  A bidimensional array containing the input data from MNIST 
     * test data set.
     */
    public static double[][] getTestDataIn(){
        return testDataIn;
    }
    /**
     *
     * @return  An array containing the real outputs of each example in MNIST 
     * test data set.
     */
    public static double[] getTestDataOut(){
        return testDataOut;
    }
    
    // Size getters
    /**
     *
     * @return  Size of MNIST training data set size.
     */
        public static int getTrainingDataSize(){
        return trainingDataIn.length;
    }
    /**
     *
     * @return  Size of each input example in MNIST training data size.
     */
    public static int getInputSize(){
        return trainingDataIn[0].length; 
    }
    /**
     *
     * @return  Size of MNIST validation data set size.
     */
    public static int getValidationDataSize(){
        return validationDataIn.length;
    }
    /**
     *
     * @return  Size of MNIST test data set size.
     */
    public static int getTestDataSize(){
        return testDataIn.length;
    }

}
