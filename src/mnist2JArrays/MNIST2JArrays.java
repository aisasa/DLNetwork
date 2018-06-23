package mnist2JArrays;

import java.io.*;
import java.util.Scanner;
import java.util.zip.*;

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
    
    public static void mnist2JArrays() throws IOException, ClassNotFoundException {
        // Training data
        trainingDataIn = new double[TRAINING_DATA_SIZE][INPUT_SIZE];
        try (Scanner scan = new Scanner(new File("trainingDataIn.txt"))) {
            for (int i = 0; i < TRAINING_DATA_SIZE; i++) 
                for (int j = 0; j < INPUT_SIZE; j++) 
                    trainingDataIn[i][j] = Double.parseDouble(scan.findWithinHorizon("0\\.[0-9]*", INPUT_SIZE));
        }
        try (ObjectOutputStream out = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream("trainingIn.dat")))) {
            out.writeObject(trainingDataIn);
        }
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
        try (ObjectOutputStream out = new ObjectOutputStream(
                new GZIPOutputStream(new FileOutputStream("trainingOut.dat")))) {
            out.writeObject(trainingDataOut);
        }

        // Validation data
        validationDataIn = new double[VALIDATION_DATA_SIZE][INPUT_SIZE];
        try (Scanner scan = new Scanner(new File("validationDataIn.txt"))) {
            for (int i = 0; i < VALIDATION_DATA_SIZE; i++) 
                for (int j = 0; j < INPUT_SIZE; j++) 
                    validationDataIn[i][j] = Double.parseDouble(scan.findWithinHorizon("0\\.[0-9]*", INPUT_SIZE));
        }
        try (ObjectOutputStream out = new ObjectOutputStream(
                new GZIPOutputStream(new FileOutputStream("validationIn.dat")))) {
            out.writeObject(validationDataIn);
        }
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
        try (ObjectOutputStream out = new ObjectOutputStream(
                new GZIPOutputStream(new FileOutputStream("validationOut.dat")))) {
            out.writeObject(validationDataOut);
        }

        // Test data
        testDataIn = new double[TEST_DATA_SIZE][INPUT_SIZE];
        try (Scanner scan = new Scanner(new File("testDataIn.txt"))) {  // testDataIn.txt
            for (int i = 0; i < TEST_DATA_SIZE; i++) 
                for (int j = 0; j < INPUT_SIZE; j++) 
                    testDataIn[i][j] = Double.parseDouble(scan.findWithinHorizon("0\\.[0-9]*", INPUT_SIZE));
        }
        try (ObjectOutputStream out = new ObjectOutputStream(
                new GZIPOutputStream(new FileOutputStream("testIn.dat")))) {
            out.writeObject(testDataIn);
        }
        testDataOut = new double[TEST_DATA_SIZE];
        try (FileReader input = new FileReader("testDataOut.txt")) {   // testDataOut.txt
            int c;
            for (int i = 0; i < TEST_DATA_SIZE; i++) {
                c = input.read();
                while (c == '\n') 
                    c = input.read();
                testDataOut[i] = Character.getNumericValue(c);
            }
        }
        try (ObjectOutputStream out = new ObjectOutputStream(
                new GZIPOutputStream(new FileOutputStream("testOut.dat")))) {
            out.writeObject(testDataOut);
        }

    }
}
