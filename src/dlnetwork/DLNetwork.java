package dlnetwork;

import java.io.*;                           // Write and read files
import java.util.*;                         // ArrayList and Arrays
import java.util.zip.*;                     // Zip and unzip files

public class DLNetwork {
    // Constants
    static final int MIN_SCORE_REF = 9840;  // Min. success rate to save weights and biases
    static final double MIN_LRN_R = 0.0001; // Min. learn rate
    static final double ERR_THR = 0.035;    // Error threshold to activate adapt. learning rate
    // Parameters
    protected int[] netShape;               // Neural network structure
    protected int nLayers;                  // # of layers
    protected double learnRate;             // Learning rate
    public static enum CostFn{QUADRATIC, CROSS_ENTROPY};   // Cost functions
    protected CostFn costFunction;
    public static enum Reglz{NO, L2};       // Regularization type, if any
    protected Reglz regularization;
    protected double lambda;                // L2 regularization lambda parameter
    public static enum AdaptLRate{NO, SQRT, QUAD, LIN}; // Adaptative learning, if any
    protected AdaptLRate adaptLearnRate;
    protected double linSlope;              // Slope of line equation for LIN     
    public DLInit.InitType initWBType;      // Initialice or load weights/biases?
    public int bestSuccessRate;             // Best score between successes
    public boolean saveBest = true;         // Save the best score?
    // Variables
    protected double[][] x;                 // Input array
    protected double[] real;                // Real results to compare
    protected ArrayList <double[][]> w;     // List of weights' matrices     
    protected ArrayList <double[]> b;       // List of biases' arrays
    protected ArrayList <double[][]> z;     // List of zs' (linear results) arrays
    protected ArrayList <double[][]> y;     // List of outputs' (sigmoided) arrays
    protected ArrayList <double[][]> deltas;// Backpropagated errors
    protected ArrayList <double[][]> gradW; // Gradient weights
    protected ArrayList <double[][]> gradB; // Gradient biases
   
    DLNetwork(int[] shape, double lr,  CostFn cFn, Reglz reg, double lmbd, AdaptLRate adLR, DLInit.InitType init) 
            throws IOException, ClassNotFoundException{
        netShape = shape;                       
        nLayers = netShape.length;
        learnRate = lr;
        costFunction = cFn;
        regularization = reg;
        lambda = lmbd;
        // Adaptative learning rate 
        adaptLearnRate = adLR;              
        if(adaptLearnRate == AdaptLRate.LIN)    // If lineal: newLearnRate = slope*error + errorMin
            linSlope = (learnRate-MIN_LRN_R)/ERR_THR;   // First compute slope
        // Initatilizing weights, biases and others
        bestSuccessRate = MIN_SCORE_REF;
        initWBType = init;
        w = DLInit.initW(initWBType, netShape);
        b = DLInit.initB(initWBType, netShape);
        z = new ArrayList(nLayers - 1);     // Inputs' arrays in each layer (except l1) 
        y = new ArrayList(nLayers);         // Activations: y = sigmoid(z)
    }
    
    private void feedForward(double[][] in){ // 'in' initialized with x
        // For each layer (input layer not included):
        for(int i=0; i<nLayers-1; i++){
            // 1. Compute dot product in by matrix w
            in = DLMath.dotProd(w.get(i), in);
            // 2. Add bias
            for(int j=0; j<in.length; j++)
                in[j][0] += b.get(i)[j];
            // 3. Include linear operations array in z and sigmoid array in y
            double[][] in_copy = new double[in.length][in[0].length];
            System.arraycopy(in, 0, in_copy, 0, in.length);
            z.add(in_copy);
            y.add(in = DLMath.sigmoid(in));
            // 4. And reuse new 'in' values as new input for next layer
        }
    }
    
    private void computeError(){
        // Computing error in last layer L (* operator is Hadamard product):  
        // delta_L = (dC/dy_L)*sigmoid_deriv(z_L) = (y_L-real)*sigmoid_deriv(z_L)
        deltas = new ArrayList(nLayers-1);
        double[][] delta_L = new double[netShape[nLayers-1]][1];
        // (dC/dy_L) = (y_L-real)
        double[] cost_d = DLMath.costQuadDeriv(DLMath.vTranspose(y.get(y.size()-1)), real);
        // sigmoid_deriv(z_L)
        double[][] sigm_d = DLMath.sigmoidDeriv(z.get(z.size()-1));
        // Hadamard product with cost function election
        for(int i=0; i<netShape[nLayers-1]; i++)
            if(costFunction == CostFn.QUADRATIC)
                delta_L[i][0] = cost_d[i]*sigm_d[i][0];
            else if(costFunction == CostFn.CROSS_ENTROPY)
                delta_L[i][0] = cost_d[i]; 
        // Last layer deltas array stored in deltas list
        deltas.add(delta_L);    
    }
    
    private void backProp(){ 
        // Backpropagating errors from L layer through previous layers l back:
        // delta_l = (w_l+1)dotProduct(delta_l+1)*sigmoid_deriv(z_l)
        for(int i=nLayers-2; i>0; i--){
            // (w_l+1)dotProduct(delta_l+1)
            double[][] dp = DLMath.dotProd(DLMath.mTranspose(w.get(i)), deltas.get(0));
            // sigmoid_deriv(z_l)
            double[][] sd = DLMath.sigmoidDeriv(z.get(i-1));
            // Hadamard product
            double[][] delta_l = new double[z.get(i-1).length][1];
            for(int j=0; j<z.get(i-1).length; j++)
                delta_l[j][0] = dp[j][0]*sd[j][0];
            // Adding new deltas array in deltas list
            deltas.add(0, delta_l);
        }
    }
    
    private void computeGradient(){   
        // How cost changes as weights and biases change:
        // dC/dw_l = (y_l-1)dotProduct(delta_l); dC/db_l = delta_l
        gradW = new ArrayList(nLayers-1);
        gradB = new ArrayList(nLayers-1);
        // dC/dw_l
        for(int i=0; i<nLayers-1; i++)
            gradW.add(DLMath.dotProd(deltas.get(i), DLMath.mTranspose(y.get(i)))); 
        // dC/db_l
        gradB = deltas;
    }
    
    private void gradDescent(int mb){  
        // Updating weights and biases (note averaged gradients by /mb):
        // First compute regularization parameter factor. 
        double regParam = 1.0;
        if(regularization == Reglz.NO);     // Do nothing, regParam = 1
        else if(regularization == Reglz.L2) // If L2, 1-learnRate*(lambda/trainingSize):
            regParam = 1-learnRate*(lambda/MNISTStore.getTrainingDataSize());   //getTrainingDataSize());
        // w_l = w_l - learningRate*dC/dw_l (see computeGradient() for dC/dw_l)
        for(int i=0; i<w.size(); i++)
            for(int j=0; j<w.get(i).length; j++)
                for(int k=0; k<w.get(i)[0].length; k++)
                    w.get(i)[j][k] = regParam * w.get(i)[j][k] - (learnRate * gradW.get(i)[j][k]/mb);
        // b_l = b_l - learningRate*dC/db_l (see computeGradient() for dC/db_l)
        for(int i=0; i<w.size(); i++)
            for(int j=0; j<w.get(i).length; j++)
                b.get(i)[j] = b.get(i)[j] - (learnRate * gradB.get(i)[j][0]/mb);
    }
    
    public void doSGD(int mb) throws IOException{  // Mini-batch as parameter
        // Stochastic Gradient Descent process:
        ArrayList<double[][]> gradWTemp = new ArrayList(nLayers-1);
        ArrayList<double[][]> gradBTemp = new ArrayList(nLayers-1);
        // For the entire training data set, taken by mini-batch subsets...
        for(int i=0; i<MNISTStore.getTrainingDataSize(); i=i+mb){ 
            // ...and for each example in each mini-batch
            for(int j=0; j<mb; j++){
                // ...load its input set in x array...
                for(int k=0; k<MNISTStore.getInputSize(); k++)
                    x[k][0] = MNISTStore.getTrainingDataIn()[i+j][k]; //+ (Math.random()/noiseDivisor); //noise;  // <======= Added noise!!!
                // ...and include it as first 'y' as needed later in computeGradient()
                y.add(x);    
                // Then load the real result linked to the example
                real = DLMath.getOutputVector((int)MNISTStore.getTrainingDataOut()[i+j]);
                // Compute the SGD process as learning algorithm...
                feedForward(x);             // Feed forwarding
                computeError();             // Output error
                backProp();                 // Backpropagation
                computeGradient();          // Gradient
                // ...taking into account the subset of examples in mini-batch...
                if(j==0){                   // Initializing gradient with previous values
                    gradWTemp = new ArrayList(gradW);   
                    gradBTemp = new ArrayList(gradB);  
                }
                else{                       // Adding each new value in mbatch process
                    gradWTemp = DLMath.addMatrInLists(gradWTemp, gradW);  
                    gradBTemp = DLMath.addMatrInLists(gradBTemp, gradW);  
                }
                // ...to compute accumulated gradient matrices (averaged in gradDescent())
                gradW = gradWTemp;   
                gradB = gradBTemp;
            }
            // Finally, after each mini-batch update weights/biases with gradients...
            gradDescent(mb);                   
            // ...and restart zs and ys to be ready for a new epoch.
            z = new ArrayList(nLayers - 1);     
            y = new ArrayList(nLayers);        
        }
        
        /*// Stochastic Gradient Descent simplified first code for fixed mini-batch = 1  
        for(int i=0; i<MNISTloader.getTrainingDataSize(); i++){     
            for(int k=0; k<MNISTloader.getInputSize(); k++)
                x[k][0] = MNISTloader.getTrainingDataIn()[i][k];
            y.add(x);                       // Add input as first y
            real = DLMath.getOutputVector((int)MNISTloader.getTrainingDataOut()[i]);
            feedForward(x);                 // Feed forwarding
            computeError();                 // Output error
            backProp();                     // Backpropagation
            computeGradient();              // Gradient
            gradDescent();                  // Gradient descent
            z = new ArrayList(nLayers - 1); // Restart zs
            y = new ArrayList(nLayers);     // Restart ys
        }*/
    }
    
    public double doTest() throws IOException{
        // Time to confirm network goodness:
        int success = 0;                   // Success marker
        // Loading each example from test data set,...
        for(int i=0; i<MNISTStore.getTestDataSize(); i++){     
            for(int j=0; j<MNISTStore.getInputSize(); j++)
                // ...both input...
                x[j][0] = MNISTStore.getTestDataIn()[i][j];    // get[Test|Validation]DataIn
            y.add(x);                           // Add input as first y
            // ...and tied real result
            real = DLMath.getOutputVector((int)MNISTStore.getTestDataOut()[i]);    // get[Test|Validation]DataOut
            // Do feed forward
            feedForward(x);               
            // Take result of network, y, treat it,...
            double[] y_t = DLMath.vTranspose(y.get(y.size()-1));    
            // ...and convert to a comparable format with real result
            int maxIndex = 0;
            for(int j=0; j<y_t.length; j++)
                maxIndex = y_t[j] > y_t[maxIndex] ? j : maxIndex;
            y_t = DLMath.getOutputVector(maxIndex);
            // Finally compare both results and accumulate successes
            if(Arrays.equals(y_t, real))
                success++;
        }
        // Once test done, update learning rate according to error obtained...
        if(adaptLearnRate != AdaptLRate.NO){
            double percent = success*100/MNISTStore.getTestDataSize();
            updateLearnRate((100.0 - percent)/100.0);   // Error = (100 - percent of success)/100
        }
        // ...and save if best model
        if(saveBest && (success > bestSuccessRate)){
            bestSuccessRate = success;
            if(saveModel(bestSuccessRate))
                System.out.println("Saved w and b lists as best scored: "); 
        }
        return success;
    }
    
    private void updateLearnRate(double error){
        if((adaptLearnRate == AdaptLRate.LIN) && (error < ERR_THR))
            // newLearnRate = slope*error + minLearnRate, where...
            // ...slope = (initLearnRate-minLearnRate)/errorActivationThreshold
            learnRate = linSlope * error + MIN_LRN_R;
        /*else if(adaptLearnRate == AdaptLRate.SQRT)  

        else if(adaptLearnRate == AdaptLRate.QUAD)*/

    }
    
    public boolean saveModel(int success) throws IOException{
        try(ObjectOutputStream out = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream("ws" + success + ".dat")))){
                out.writeObject(w);
            }
        try(ObjectOutputStream out = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream("bs" + success + ".dat")))){
                out.writeObject(b);
            }
        catch(IOException e){
            return false;
        }
        return true;
    }
    
}