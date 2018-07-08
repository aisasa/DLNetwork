package dlnetwork;

import java.util.*;                         // ArrayList and Arrays
import java.io.*;                           // IOException, I/O streams
import java.util.zip.*;                     // Zip/unzip files

/**
 * A set of different procedures to initialize the weights and biases sets in a
 * deep learning network.
 * 
 * @author  Agustin Isasa Cuartero
 * @version 0.9
 */
public class DLInit {
    // Constants
    protected final static double RND_SUBT = 0.5;   // Random subtraction -> [-0.5, 0.5]
    protected final static double RND_DIV = 1.0;    // Random divisor
    // Var
    private static ArrayList<ArrayList> wbArrays; // Weights and biases arrays in a container
    // Clean init | Clean init & save | Load previous saved random set | Load a set by console
    /**
     * Enum with the different initialization types of elements in weights and 
     * biases sets. Type can be:
     * <ul>
     * <li>RANDOM: initial random results in (0, 1) range, modified then by 
     * RND_SUBT and/or RND_DIV.
     * <li>RAND_AND_SAVE: same that RANDOM but recording weights and biases to 
     * reuse in performance comparations, if needed.
     * <li>LOAD_PRE_SAVED: directly load of last weights and biases sets 
     * previously saved with RAND_AND_SAVE.
     * <li>LOAD_BY_NAME: load manually (by console dialogue) weights and 
     * biases sets previously saved (for instance, with the record from a 
     * minimum score option).
     * </ul>
     * 
     */
    protected static enum WBInitType{RANDOM, RAND_AND_SAVE, LOAD_PRE_SAVED, LOAD_BY_NAME};
    protected static WBInitType wbInitType;
    protected static String wFileName;
    protected static String bFileName;
    
    /**
     * Entry method to some different ways to initialize weights and biases
     * sets of a deep learning network.
     * 
     * @param initT     Type of initialization. It can be:
     *                  <ul>
     *                  <li>RANDOM: initial random results in (0, 1) range, 
     *                  modified then by RND_SUBT and/or RND_DIV.
     *                  <li>RAND_AND_SAVE: same that RANDOM but recording weights
     *                  and biases to reuse in performance comparations.
     *                  <li>LOAD_PRE_SAVED: directly load of last weights and 
     *                  biases sets previously saved with RAND_AND_SAVE.
     *                  <li>LOAD_BY_NAME: load manually (by console dialogue) 
     *                  weights and biases sets previously saved (for instance,
     *                  with the record from a minimum score option).
     *                  </ul>
     * @param netShape  Network shape in a vector form, that is, in which each 
     *                  element is the size of its respective layer.
     * @return          An array list containing both sets of initialized 
     *                  weights and biases.
     * @throws          java.io.IOException                 If file does not exist
     * @throws          java.lang.ClassNotFoundException    If array does not exist in saved files
     */
    public static ArrayList<ArrayList> initWB(WBInitType initT, int[] netShape) throws IOException, ClassNotFoundException{
        wbInitType = initT;
        wbArrays = new ArrayList<>();
        ArrayList <double[][]> w = new ArrayList(netShape.length - 1);
        ArrayList <double[][]> b = new ArrayList(netShape.length - 1);  
        if((wbInitType == WBInitType.RANDOM) || (wbInitType == WBInitType.RAND_AND_SAVE)){
            w = _initW(netShape);  
            b = _initB(netShape);
        }
        else if(wbInitType == WBInitType.LOAD_PRE_SAVED){
            w = _loadW("");                 // [initial_W.dat | ws9XXX....dat]
            b = _loadB("");                 // [initial_B.dat | bs9XXX....dat]
        }
        else if(wbInitType == WBInitType.LOAD_BY_NAME){
            System.out.println("Enter weights file name: ");
            w = _loadW(wFileName = new Scanner(System.in).nextLine());
            System.out.println("Enter biases file name: ");
            b = _loadB(bFileName = new Scanner(System.in).nextLine());
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
                    u[j][k] = (Math.random()-RND_SUBT)/RND_DIV;  
            w.add(u);
        }
        // Saving the initialization weights for future reference in tests, if chosen
        if(wbInitType == WBInitType.RAND_AND_SAVE){
            try(ObjectOutputStream out = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream("initial_W.dat")))){
                    out.writeObject(w);
                    out.close();
                }
            System.out.println("Initial weights data saved");
        }
        return w;
    }
    
    private static ArrayList<double[][]> _initB(int[] netShape) throws IOException, ClassNotFoundException{
        // Initializing network biases           
        int nLayers = netShape.length;
        ArrayList <double[][]> b = new ArrayList(nLayers - 1);    // List of biases' arrays
        double[][] v;
        for(int i=1; i<nLayers; i++){
            v = new double[1][netShape[i]];
            for(int j=0; j<netShape[i]; j++)
                v[0][j] = (Math.random()-RND_SUBT)/RND_DIV; 
            b.add(v);            
        } 
        // Saving the initialization biases for future reference in tests, if chosen
        if(wbInitType == WBInitType.RAND_AND_SAVE){
            try(ObjectOutputStream out = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream("initial_B.dat")))){
                    out.writeObject(b);
                    out.close();
                }
            System.out.println("Initial biases data saved");
        }
        return b;
    }
    
    private static ArrayList<double[][]> _loadW(String fName) throws IOException, ClassNotFoundException{
        if(fName.equals(""))
            wFileName = "initial_W.dat";     
        ArrayList <double[][]> w;   
        try(ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(new FileInputStream(wFileName)))){ 
                w = (ArrayList<double[][]>)in.readObject();
            }
        return w;
    }
    
    private static ArrayList<double[][]> _loadB(String fName) throws IOException, ClassNotFoundException{
        if(fName.equals(""))
            bFileName = "initial_B.dat";     
        ArrayList <double[][]> b;
        try(ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(new FileInputStream(bFileName)))){ 
                b = (ArrayList<double[][]>)in.readObject();
            }
        return b;
    }
    
}