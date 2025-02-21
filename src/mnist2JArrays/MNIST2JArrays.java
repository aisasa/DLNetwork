package mnist2JArrays;

import java.io.*;                           // IOException, I/O streams, File
import java.util.ArrayList;                 // That same
import java.util.Scanner;                   // Reading txt files
import java.util.zip.*;                     // Zip/unzip files

/**
 * Convert MNIST text data sets to Java arrays.
 * 
 * @author  Agustin Isasa Cuartero
 * @version 0.9 
 */
public class MNIST2JArrays {
    // Constants
    private static final int TRAINING_DATA_SIZE = 50000;
    private static final int INPUT_SIZE = 784;
    private static final int VALIDATION_DATA_SIZE = 10000;
    private static final int TEST_DATA_SIZE = 10000;
    // Data structures
    private static double[][] trainingDataIn;
    private static double[] trainingDataOut;
    private static double[][] validationDataIn;
    private static double[] validationDataOut;
    private static double[][] testDataIn;
    private static double[] testDataOut;
    private static final ArrayList trainingData = new ArrayList();
    private static final ArrayList validationData = new ArrayList();
    private static final ArrayList testData = new ArrayList();
    
    
    /**
     * Static method that scans the MNIST text data sets and convert them to 
     * Java arrays, saving finally data in files. 
     * 
     * @throws IOException      If file does not exist.
     */
    public static void mnist2JArrays() throws IOException{ 
        // Training data
        trainingDataIn = new double[TRAINING_DATA_SIZE][INPUT_SIZE];
        try (Scanner scan = new Scanner(new File("trainingDataIn.txt"))) {
            for (int i = 0; i < TRAINING_DATA_SIZE; i++) 
                for (int j = 0; j < INPUT_SIZE; j++) 
                    trainingDataIn[i][j] = Double.parseDouble(scan.findWithinHorizon("0\\.[0-9]*", INPUT_SIZE));
        }
        trainingData.add(trainingDataIn);
        trainingDataOut = new double[TRAINING_DATA_SIZE];
        try (FileReader input = new FileReader("trainingDataOut.txt")) {
            int c;
            for (int i = 0; i < TRAINING_DATA_SIZE; i++) {
                c = input.read();
                while (c == '\n') 
                    c = input.read();
                trainingDataOut[i] = Character.getNumericValue(c);
            }
        }
        trainingData.add(trainingDataOut);
        try (ObjectOutputStream out = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream("training.dat")))) {
            out.writeObject(trainingData);
            out.close();
        }

        // Validation data
        validationDataIn = new double[VALIDATION_DATA_SIZE][INPUT_SIZE];
        try (Scanner scan = new Scanner(new File("validationDataIn.txt"))) {
            for (int i = 0; i < VALIDATION_DATA_SIZE; i++) 
                for (int j = 0; j < INPUT_SIZE; j++) 
                    validationDataIn[i][j] = Double.parseDouble(scan.findWithinHorizon("0\\.[0-9]*", INPUT_SIZE));
        }
        validationData.add(validationDataIn);
        validationDataOut = new double[VALIDATION_DATA_SIZE];
        try (FileReader input = new FileReader("validationDataOut.txt")) {
            int c;
            for (int i = 0; i < VALIDATION_DATA_SIZE; i++) {
                c = input.read();
                while (c == '\n') 
                    c = input.read();
                validationDataOut[i] = Character.getNumericValue(c);
            }
        }
        validationData.add(validationDataOut);
        try (ObjectOutputStream out = new ObjectOutputStream(
                new GZIPOutputStream(new FileOutputStream("validation.dat")))) {
            out.writeObject(validationData);
        }

        // Test data
        testDataIn = new double[TEST_DATA_SIZE][INPUT_SIZE];
        try (Scanner scan = new Scanner(new File("testDataIn.txt"))) {  
            for (int i = 0; i < TEST_DATA_SIZE; i++) 
                for (int j = 0; j < INPUT_SIZE; j++) 
                    testDataIn[i][j] = Double.parseDouble(scan.findWithinHorizon("0\\.[0-9]*", INPUT_SIZE));
        }
        testData.add(testDataIn);
        testDataOut = new double[TEST_DATA_SIZE];
        try (FileReader input = new FileReader("testDataOut.txt")) {   
            int c;
            for (int i = 0; i < TEST_DATA_SIZE; i++) {
                c = input.read();
                while (c == '\n') 
                    c = input.read();
                testDataOut[i] = Character.getNumericValue(c);
            }
        }
        testData.add(testDataOut);
        try (ObjectOutputStream out = new ObjectOutputStream(
                new GZIPOutputStream(new FileOutputStream("test.dat")))) {
            out.writeObject(testData);
        }
    }
}
