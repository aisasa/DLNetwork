package dlnetwork;

import java.io.*;                           // Write/read files
import java.util.*;                         // ArrayList and Arrays
import java.util.zip.*;                     // Zip/unzip files

public class DLNetwork {
    // Constants
    static final double MIN_LRN_R = 0.0001; // Min. learn rate
    static final double ERR_THR = 0.035;    // Error threshold to activate adapt. learning rate 
    // Parameters
    protected int[] netShape;               // Neural network structure
    protected int nLayers;                  // # of layers
    protected double learnRate;             // Learning rate
    protected int miniBatch;                // Subset of the training set. Here, the lower the better
    public static enum CostFn{QUADRATIC, CROSS_ENTROPY};   // Cost functions
    protected CostFn costFunction;
    public static enum Reglz{NO, L2};       // Regularization type, if any
    protected Reglz regularization;
    protected double lambda;                // L2 regularization lambda parameter
    public static enum AdaptLRate{NO, LIN, QUAD, SQRT}; // Adaptive learning, if any
    protected AdaptLRate adaptLearnRate;
    protected double linSlope;              // Slope of line equation for LIN     
    // Variables (all vectors treated as matrices with one row OR one column)
    protected double[][] x;                 // Input array                        
    protected double[] realOut;             // Real result linked to an entry
    protected ArrayList <double[][]> w;     // List of weights' matrices     
    protected ArrayList <double[][]> b;     // List of biases' arrays      
    protected ArrayList <double[][]> z;     // List of zs' (linear results) arrays  
    protected ArrayList <double[][]> y;     // List of outputs' (sigmoided) arrays  
    protected ArrayList <double[][]> deltas;// Backpropagated errors arrays               
    protected ArrayList <double[][]> gradW; // Gradient weights arrays                    
    protected ArrayList <double[][]> gradB; // Gradient biases arrays                      
    // Execution layout
    protected int epochs;                   // How many times we treat the entire training data set
    protected boolean shuffleSets;          // Shuffle or not training sets between epochs?
    protected boolean saveBest = false;     // Save the best score model? (from minScoreRef)
    protected int minScoreRef;              // Minimum score from which to record best results
    protected int bestSuccessRate;          // Best score between successes in an epoch 
    
    DLNetwork(int[] shape, double lr, CostFn cFn, Reglz reg, double lmbd, AdaptLRate adLR, int mBatch, DLInit.WBInitType initT) 
            throws IOException, ClassNotFoundException{
        // Network layout
        netShape = shape;                       
        nLayers = netShape.length;
        learnRate = lr;
        costFunction = cFn;
        regularization = reg;
        lambda = lmbd;
        // Adaptative learning rate 
        adaptLearnRate = adLR;              
        if(adaptLearnRate == AdaptLRate.LIN) // If lineal: newLearnRate = slope*error + errorMin
            linSlope = (learnRate-MIN_LRN_R)/ERR_THR;   // First compute slope
        miniBatch = mBatch;
        // Initatilizing weights and biases 
        ArrayList<ArrayList> wbContainer = DLInit.initWB(initT, netShape);
        w = wbContainer.get(0); //(ArrayList<double[][]>)DLInit.wbArrays.get(0);
        b = wbContainer.get(1); //(ArrayList<double[]>)DLInit.wbArrays.get(1);
        // Initializing variables
        z = new ArrayList(nLayers - 1);     // Inputs' arrays in each layer (except l1) 
        y = new ArrayList(nLayers);         // Activations: y = sigmoid(z)
        x = new double[MNISTStore.getInputSize()][1];   // Array x contains each training example input   
    }
    
    public void start(int epch, boolean shuffle) throws IOException{
        epochs = epch;
        shuffleSets = shuffle;
        System.out.println("Starting computation: "); 
        System.out.println("· Epochs: " + epochs);
        System.out.println("· Shuffle: " + shuffleSets);
        System.out.println("· Record best results model: " + saveBest);
        if(saveBest)
            System.out.println("    Record from success: " + minScoreRef);
        System.out.println("· Start time: " + new java.util.Date());
        // For the entire training set each time, and for several times (epochs):
        for(int i =0; i<epochs; i++){
            // 1. Go SGD passing an appropriate size of minibatch
            doSGD(miniBatch);
            // 2. Then shuffling the set if provided
            if(shuffleSets)
                MNISTStore.shuffleMNIST();
            // 3. Testing results after each epoch
            System.out.println("Epoch " + i + ": " + doTest()*100/10000 + "%");  
            // 4. And go for another one.
        }
        System.out.println("End computation: " + new java.util.Date());
    }
    // If save best scored model, start() admits minimum score to start record from
    public void start(int epch, boolean shuffle, int minScore) throws IOException{
        saveBest = true;   
        minScoreRef = minScore;
        bestSuccessRate = minScoreRef;
        this.start(epch, shuffle);
    }
    
    protected void doSGD(int mb) throws IOException{  // Mini-batch as parameter
        // Stochastic Gradient Descent process:
        ArrayList<double[][]> gradWTemp = new ArrayList(nLayers-1);
        ArrayList<double[][]> gradBTemp = new ArrayList(nLayers-1);
        // For the entire training data set, taken by mini-batch subsets...
        for(int i=0; i<MNISTStore.getTrainingDataSize(); i=i+mb){ 
            // ...and for each example in each mini-batch:
            for(int j=0; j<mb; j++){
                // 1. Load its input set in x array...
                for(int k=0; k<MNISTStore.getInputSize(); k++)
                    x[k][0] = MNISTStore.getTrainingDataIn()[i+j][k]; 
                // ...and include it as first 'y' as needed later in computeGradient()
                y.add(x);    
                // 2. Load the real result linked to the example
                realOut = DLMath.getOutputVector((int)MNISTStore.getTrainingDataOut()[i+j]);
                // 3. Compute the SGD process as learning algorithm...
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
            // 4. After each mini-batch update weights/biases with gradients...
            gradDescent(mb);                   
            // 5. Restart zs and ys to be ready for a new epoch.
            z = new ArrayList(nLayers - 1);     
            y = new ArrayList(nLayers);        
        }
        
        /*// Stochastic Gradient Descent minimal first code for mini-batch = 1  
        for(int i=0; i<MNISTloader.getTrainingDataSize(); i++){     
            for(int k=0; k<MNISTloader.getInputSize(); k++)
                x[k][0] = MNISTloader.getTrainingDataIn()[i][k];
            y.add(x);                       // Add input as first y
            realOut = DLMath.getOutputVector((int)MNISTloader.getTrainingDataOut()[i]);
            feedForward(x);                 // Feed forwarding
            computeError();                 // Output error
            backProp();                     // Backpropagation
            computeGradient();              // Gradient
            gradDescent();                  // Gradient descent
            z = new ArrayList(nLayers - 1); // Restart zs
            y = new ArrayList(nLayers);     // Restart ys
        }*/
    }
    
    protected double doTest() throws IOException{
        // Time to confirm network goodness:
        int success = 0;                    // Success counter
        // 1. Loading each example from test data set,...
        for(int i=0; i<MNISTStore.getTestDataSize(); i++){     
            for(int j=0; j<MNISTStore.getInputSize(); j++)
                // ...both input...
                x[j][0] = MNISTStore.getTestDataIn()[i][j];    // get[Test|Validation]DataIn
            y.add(x);                       // Add input as first y
            // ...and tied realOut result
            realOut = DLMath.getOutputVector((int)MNISTStore.getTestDataOut()[i]); // get[Test|Validation]DataOut
            // 2. Do feed forward
            feedForward(x);               
            // 3. Take result of network, y, treat it,...
            double[] y_t = DLMath.vTranspose(y.get(y.size()-1));    
            // ...and convert to a comparable format with realOut result
            int maxIndex = 0;
            for(int j=0; j<y_t.length; j++)
                maxIndex = y_t[j] > y_t[maxIndex] ? j : maxIndex;
            y_t = DLMath.getOutputVector(maxIndex);
            // 4. Finally compare both results and accumulate successes
            if(Arrays.equals(y_t, realOut))
                success++;
        }
        // 5. Once test done, update learning rate according to error obtained
        if(adaptLearnRate != AdaptLRate.NO){
            double percent = success*100.0/(double)MNISTStore.getTestDataSize();
            updateLearnRate((100.0 - percent)/100.0);   // Error = (100 - percent of success)/100
        }
        // 6. Save if best model
        if(saveBest && (success > bestSuccessRate)){
            bestSuccessRate = success;
            if(saveModel(bestSuccessRate))
                System.out.println("Saved w and b lists as best scored: "); 
        }
        return success;
    }
    
    private void feedForward(double[][] in){ // 'in' initialized with x
        // For each layer (input layer not included):
        for(int i=0; i<nLayers-1; i++){
            // 1. Compute dot product in by matrix w
            in = DLMath.dotProd(w.get(i), in);
            // 2. Add bias
            for(int j=0; j<in.length; j++)
                in[j][0] += b.get(i)[0][j];
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
        // delta_L = (dC/dy_L)*sigmoid_deriv(z_L) = (y_L-realOut)*sigmoid_deriv(z_L)
        deltas = new ArrayList(nLayers-1);
        double[][] delta_L = new double[netShape[nLayers-1]][1];
        // (dC/dy_L) = (y_L-realOut)
        double[] cost_d = DLMath.costQuadDeriv(DLMath.vTranspose(y.get(y.size()-1)), realOut);
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
                b.get(i)[0][j] = b.get(i)[0][j] - (learnRate * gradB.get(i)[j][0]/mb);
    }
    
    private void updateLearnRate(double error){
        if(error < ERR_THR){
            if(adaptLearnRate == AdaptLRate.LIN)
                // newLearnRate = slope*error + minLearnRate, where...
                learnRate = linSlope * error + MIN_LRN_R;
                // ...slope = (initLearnRate-minLearnRate)/errorActivationThreshold
            else if(adaptLearnRate == AdaptLRate.QUAD)
                learnRate = error*error/ERR_THR;
            else if(adaptLearnRate == AdaptLRate.SQRT)
                // newLearnRate = sqrt(sqrt(error * ERR_THR^3))
                learnRate = Math.pow(error * Math.pow(ERR_THR, 3), 1.0/4.0);
        }
    }
    
    public boolean saveModel(int success) throws IOException{
        try{
            ObjectOutputStream out = 
                    new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream("ws" + success + "-" + Arrays.toString(netShape) + ".dat")));
            out.writeObject(w);
            out = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream("bs" + success + "-" + Arrays.toString(netShape) + ".dat")));
            out.writeObject(b);
            }
        catch(IOException e){
            return false;
        }
        return true;
    }
    
    public void paramRef(){
        // Network parameters reference
        System.out.println("Network reference: ");
        System.out.println("· Shape: " + Arrays.toString(netShape));
        System.out.println("· Cost function: " + costFunction);
        System.out.println("· Learning rate: " + learnRate);
        System.out.println("· Adaptive learning rate: " + adaptLearnRate);
        if(adaptLearnRate != DLNetwork.AdaptLRate.NO) {
            System.out.println("    Activation error threshold: " + DLNetwork.ERR_THR);
            System.out.println("    Minimum learn rate: " + DLNetwork.MIN_LRN_R);
        }
        System.out.println("· Regularization: " + regularization);
        if(regularization == DLNetwork.Reglz.L2)
            System.out.println("    L2 lambda: " + lambda);
        System.out.println("· Mini batch: " + miniBatch);
        System.out.println("· Initialice weights/biases: " + DLInit.wbInitType);
        if(DLInit.wbInitType == DLInit.WBInitType.LOAD_PRE_SAVED || DLInit.wbInitType == DLInit.WBInitType.LOAD_BY_NAME){
            System.out.println("    Weights file: " + DLInit.wFileName);
            System.out.println("    Biases file: " + DLInit.bFileName);
        }
        if(DLInit.wbInitType == DLInit.WBInitType.RANDOM || DLInit.wbInitType == DLInit.WBInitType.RAND_AND_SAVE){
            System.out.println("    Random subtraction: " + DLInit.RND_SUBT);
            System.out.println("    Random divisor: " + DLInit.RND_DIV);
        }
    }
    
}