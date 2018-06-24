package dlnetwork;

import mnist2JArrays.*;                     // MNIST txt data to Java arrays
import java.io.*;                           // IOException, File, I/O streams
import java.util.zip.*;                     // Zip/unzip files

public class MNISTStore {
    private static double[][] trainingDataIn;   
    private static double[] trainingDataOut; 
    private static double[][] validationDataIn;
    private static double[] validationDataOut;
    private static double[][] testDataIn;
    private static double[] testDataOut;
    
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
    }
    
    // Shuffle data sets
    public static void shuffleMNIST(){
        DLMath.shuffle(trainingDataIn, trainingDataOut);
    }
    
    // Object getters
    public static double[][] getTrainingDataIn(){
        return trainingDataIn;
    }
    public static double[] getTrainingDataOut(){
        return trainingDataOut;
    }
    public static double[][] getValidationDataIn(){
        return validationDataIn;
    }
    public static double[] getValidationDataOut(){
        return validationDataOut;
    }
    public static double[][] getTestDataIn(){
        return testDataIn;
    }
    public static double[] getTestDataOut(){
        return testDataOut;
    }
    // Size getters
    public static int getTrainingDataSize(){
        return trainingDataIn.length;
    }
    public static int getInputSize(){
        return trainingDataIn[0].length; 
    }
    public static int getValidationDataSize(){
        return validationDataIn.length;
    }
    public static int getTestDataSize(){
        return testDataIn.length;
    }

    
}
