package dlnetwork;

import java.util.*;
import java.io.*;
import java.util.zip.*;

public class DLInit {
    // Clean initialization || Load a prev. saved random set || Load a prev. set of best scoring
    public static enum InitType{RANDOM, LOAD_PRE_SAVED, LOAD_BEST};
    
    public static ArrayList<double[][]> initW(InitType initT, int[] netShape) throws IOException, ClassNotFoundException{
        ArrayList <double[][]> w = new ArrayList(netShape.length - 1);
        if(initT == InitType.RANDOM){
            w = _initW(netShape);
        }
        else if(initT == InitType.LOAD_PRE_SAVED){
            w = _loadW();
        }
        return w;
    }
    
    public static ArrayList<double[]> initB(InitType initT, int[] netShape) throws IOException, ClassNotFoundException{
        ArrayList <double[]> b = new ArrayList(netShape.length - 1);
        if(initT == InitType.RANDOM){
            b = _initB(netShape);
        }
        else if(initT == InitType.LOAD_PRE_SAVED){
            b = _loadB();
        }
        return b;
    }
    
    public static ArrayList<double[][]> _initW(int[] netShape) throws IOException, ClassNotFoundException{ 
        // Initializing network weights
        int nLayers = netShape.length;     
        ArrayList <double[][]> w = new ArrayList(netShape.length - 1);   // List of weights' arrays
        double[][] u;
        for(int i=1; i<nLayers; i++){
            u = new double[netShape[i]][netShape[i-1]]; 
            for(int j=0; j<netShape[i]; j++)
                for(int k=0; k<netShape[i-1]; k++)
                    u[j][k] = (Math.random()-05.)/100;  // (Math.random()-0.5)/100;
            w.add(u);
        }
        // Saving the initialization weights for future reference in tests
        try(ObjectOutputStream out = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream("initial_W.dat")))){
                out.writeObject(w);
            }
        System.out.println("Initial weights data saved");
        return w;
    }
    
    public static ArrayList<double[]> _initB(int[] netShape) throws IOException, ClassNotFoundException{
        // Initializing network biases           
        int nLayers = netShape.length;
        ArrayList <double[]> b = new ArrayList(nLayers - 1);    // List of biases' arrays
        double[] v;
        for(int i=1; i<nLayers; i++){
            v = new double[netShape[i]];
            for(int j=0; j<netShape[i]; j++)
                v[j] = (Math.random()-0.5)/50; //0.5
            b.add(v);            
        } 
        // Saving the initialization biases for future reference in tests
        try(ObjectOutputStream out = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream("initial_B.dat")))){
                out.writeObject(b);
            }
        System.out.println("Initial biases data saved");
        return b;
    }
    
    public static ArrayList<double[][]> _loadW() throws IOException, ClassNotFoundException{
        String fileName = "initial_W.dat";     // [initial_W.dat | ws9858.dat]
        ArrayList <double[][]> w;   
        try(ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(new FileInputStream(fileName)))){ 
                w = (ArrayList<double[][]>)in.readObject();
            }
        return w;
    }
    
    public static ArrayList<double[]> _loadB() throws IOException, ClassNotFoundException{
        String fileName = "initial_B.dat";     // [initial_B.dat | bs9858.dat]
        ArrayList <double[]> b;
        try(ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(new FileInputStream(fileName)))){ 
                b = (ArrayList<double[]>)in.readObject();
            }
        return b;
    }
}
