package dlnetwork;

import mnist2JArrays.*;                     // MNIST txt data to Java arrays
import java.io.*;                           // IOException, File, I/O streams
import java.util.ArrayList;                 // That same   
import java.util.zip.*;                     // Zip/unzip files

/**
 * A warehouse to store MNIST Java data structures.
 * 
 * @author  Agustin Isasa Cuartero
 * @version 0.9
 */
public class MNISTStore {
    private static final ArrayList trainingData = new ArrayList();
    private static final ArrayList validationData = new ArrayList();
    private static final ArrayList testData = new ArrayList();
    
    /**
     * Transfer MNIST data from previously saved files to proper Java data
     * structures
     * 
     * @throws IOException              If file does not exist
     * @throws ClassNotFoundException   If array does not exist in saved files
     */
    public static void loadMNISTArrays() throws IOException, ClassNotFoundException{
        File fTraining = new File("training.dat");
        File fValidation = new File("validation.dat");
        File fTest = new File("test.dat");
        if(!fTraining.exists() || !fValidation.exists() || !fTest.exists())
            MNIST2JArrays.mnist2JArrays();
        
        // Training data
        try(ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(new FileInputStream(fTraining)))){ 
            ArrayList aList = (ArrayList)in.readObject();
            trainingData.add((double[][])aList.get(0));
            trainingData.add((double[])aList.get(1));
        }
        // Validation data
        try(ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(new FileInputStream(fValidation)))){ 
            ArrayList aList = (ArrayList)in.readObject();
            validationData.add((double[][])aList.get(0));
            validationData.add((double[])aList.get(1));
        }
        // Test data
        try(ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(new FileInputStream(fTest)))){ 
            ArrayList aList = (ArrayList)in.readObject();
            testData.add((double[][])aList.get(0));
            testData.add((double[])aList.get(1));
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
        DLMath.shuffle(trainingData);
    }
    
    // Object getters
     /**
     *
     * @return  A container (an array list) with two elements: a bidimensional 
     * array containing the input data from MNIST training data set and an array
     * containing the real output of each input or example in MNIST training 
     * data set.
     */
    public static ArrayList getTrainingData(){
    return trainingData;
    }
    /**
     *
     * @return  A container (an array list) with two elements: a bidimensional 
     * array containing the input data from MNIST validation data set and an 
     * array containing the corresponding real output of each input or example 
     * in MNIST validation data set.
     */
    public static ArrayList getValidationData(){
    return validationData;
    }
    /**
     *
     * @return  A container (an array list) with two elements: a bidimensional 
     * array containing the input data from MNIST test data set and an 
     * array containing the corresponding real output of each input or example 
     * in MNIST test data set.
     */
    public static ArrayList getTestData(){
    return testData;
    }
    
    // Size getters
    /**
     *
     * @return  Size of MNIST training data set size.
     */
    public static int getTrainingDataSize(){
        return ((double[][])trainingData.get(0)).length;
    }
    /**
     *
     * @return  Size of each input example in MNIST training data size.
     */
    public static int getInputSize(){
        return ((double[][])trainingData.get(0))[0].length;
    }
    /**
     *
     * @return  Size of MNIST validation data set size.
     */
    public static int getValidationDataSize(){
        return ((double[][])validationData.get(0)).length;
    }
    /**
     *
     * @return  Size of MNIST test data set size.
     */
    public static int getTestDataSize(){
        return ((double[][])testData.get(0)).length;
    }

}
