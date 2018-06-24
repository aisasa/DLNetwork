package dlnetwork;

import java.util.*;                         // ArrayList and Arrays
import java.io.*;                           // IOException, I/O streams
import java.util.zip.*;                     // Zip/unzip files

public class DLInit {
    protected static ArrayList<ArrayList> wbArrays;   // Weights and biases arrays in a container
    // Clean init || Clean init & save || Load a prev. saved random set || Load (now manually) a prev. set of best scoring
    protected static enum WBInitType{RANDOM, RAND_AND_SAVE, LOAD_PRE_SAVED, LOAD_BEST};
    protected static WBInitType wbInitType;
    
    public static ArrayList<ArrayList> initWB(WBInitType initT, int[] netShape) throws IOException, ClassNotFoundException{
        wbInitType = initT;
        wbArrays = new ArrayList<>();
        ArrayList <double[][]> w = new ArrayList(netShape.length - 1);
        ArrayList <double[]> b = new ArrayList(netShape.length - 1);
        if((wbInitType == WBInitType.RANDOM) || (wbInitType == WBInitType.RAND_AND_SAVE)){
            w = _initW(netShape);  
            b = _initB(netShape);
        }
        else if(wbInitType == WBInitType.LOAD_PRE_SAVED){
            w = _loadW();
            b = _loadB();
        }
        wbArrays.add(w);
        wbArrays.add(b);
        return wbArrays;
    }
    
    private static ArrayList<double[][]> _initW(int[] netShape) throws IOException, ClassNotFoundException{ 
        // Initializing network weights
        int nLayers = netShape.length;     
        ArrayList <double[][]> w = new ArrayList(netShape.length - 1);   // List of weights' arrays
        double[][] u;
        for(int i=1; i<nLayers; i++){
            u = new double[netShape[i]][netShape[i-1]]; 
            for(int j=0; j<netShape[i]; j++)
                for(int k=0; k<netShape[i-1]; k++)
                    u[j][k] = (Math.random()-0.5)/100;  // (Math.random()-0.5)/100;
            w.add(u);
        }
        // Saving the initialization weights for future reference in tests, if chosen
        if(wbInitType == WBInitType.RAND_AND_SAVE){
            try(ObjectOutputStream out = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream("initial_W.dat")))){
                    out.writeObject(w);
                }
            System.out.println("Initial weights data saved");
        }
        return w;
    }
    
    private static ArrayList<double[]> _initB(int[] netShape) throws IOException, ClassNotFoundException{
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
        // Saving the initialization biases for future reference in tests, if chosen
        if(wbInitType == WBInitType.RAND_AND_SAVE){
            try(ObjectOutputStream out = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream("initial_B.dat")))){
                    out.writeObject(b);
                }
            System.out.println("Initial biases data saved");
        }
        return b;
    }
    
    private static ArrayList<double[][]> _loadW() throws IOException, ClassNotFoundException{
        String fileName = "initial_W.dat";     // [initial_W.dat | ws9858.dat]
        ArrayList <double[][]> w;   
        try(ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(new FileInputStream(fileName)))){ 
                w = (ArrayList<double[][]>)in.readObject();
            }
        return w;
    }
    
    private static ArrayList<double[]> _loadB() throws IOException, ClassNotFoundException{
        String fileName = "initial_B.dat";     // [initial_B.dat | bs9858.dat]
        ArrayList <double[]> b;
        try(ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(new FileInputStream(fileName)))){ 
                b = (ArrayList<double[]>)in.readObject();
            }
        return b;
    }
}
