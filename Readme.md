# A Deep Learning Network with Java

A basic Java deep learning network to attack the MNIST data set of hand-written digits, but easily customizable for other purposes. 
Implements net shape configuration, quadratic and cross-entropy cost functions, sigmoid function in neurons, some basic adaptative 
learning rate functions, mini-batch options... As rules of the game are clearer when playing with understandable figures instead of bytes, a zipped text version of the MNIST data sets (training, validation and test) is also provided, in addition to Java arrays sets as saved files (internally zipped too).

The project is based on the first chapters of Michael Nielsen's work "Neural networks and deep learning", published on 
https://neuralnetworksanddeeplearning.com and https://github.com/mnielsen/neural-networks-and-deep-learning. My acknowledgment and 
gratitude to the author for such a magnificent job.

This project is not a direct translation between Python and Java, and keeps its own particularities 
both in the structure of files and in the program's flow. It has a few more lines of core code because of Java strongly typed
nature as well as some other causes. For instance, no external numeric library has been applied, so the necessary mathematical 
functions have been implemented in their basic incarnations in the corresponding file (DLMath.java). Don't expect, therefore, an optimized 
performance. Also it is not a good example of object oriented programming due their design fundamentals, adhered to the more algebraic 
Michael's approach instead to consider neuron abstractions.

Building this software has been one of the best ways to follow and understand the fundamentals of deep learning that Michael Nielsen 
explains in his work. 

The software works, and in particular conditions (shape {784, 100 to 800, 10}, cross-entropy cost function, with mini-batch=1 or with minimum size, learn rate = 0.12, initialization of weights with ```Math.random ()-0.5``` and biases with ```Math.random ()-0.5```) is able to obtain, in a moderate amount of epochs (say 20-40), around 98.20%-98,30%  of success in the recognition of the MNIST set, with scores in 98.50%-98.60% in less than a hundred epochs with some fortunate initialization files and parameters setup.

Use: ready to load data sets from enclosed Java arrays files. In the source directory execute DLRun.java as usual. Modify parameters (in DLRun.java) and code (surely in DLNetwork.java, ```main()```, DLInit.java, or whatever) as needed. The interface in ```main()``` (DLRun.java file) is reasonably clear and allows inmediate changes (play!) in essential parameters:

```java
public class DLRun {
    // Defining, building and starting the neural network
    public static void main(String[] args) throws IOException, ClassNotFoundException{
        // 1. Network constructor parameters
        // Defining neural net shape
        int[] initShape = {784, 200, 10}; 
        // Cost function
        DLNetwork.CostFn costF = DLNetwork.CostFn.CROSS_ENTROPY;        // QUADRATIC | CROSS_ENTROPY   
        // Init learning rate (fine tune of weights/biases variations) and possible adaptation
        double lRate = 0.12;       
        DLNetwork.AdaptLRate adapLR = DLNetwork.AdaptLRate.QUAD;        // NO | LIN | QUAD | SQRT
        // Regularization type and possible parameters
        DLNetwork.Reglz reg = DLNetwork.Reglz.L2;                       // NO | L2   
        double lambda = 1.75;                                           // L2 regularization parameter
        // How to initialize weights and biases sets
        DLInit.WBInitType wbInitType = DLInit.WBInitType.RAND_AND_SAVE; // RANDOM | RAND_AND_SAVE| LOAD_PRE_SAVED | LOAD_BY_NAME
        // Mini batch: subset of the training set. Here, the lower the better
        int miniBatch = 1;      
        
        // 2. Loading MNIST data in its arrays
        MNISTStore.loadMNISTArrays();
        
        // 3. Build neural network and show network parameters as reference
        // (int[] shape, double lr, CostFn cFn, Reglz reg, double lmbd, int mB, WBInitType init/load)
        DLNetwork net = new DLNetwork(initShape, lRate, costF, reg, lambda, adapLR, miniBatch, wbInitType); 
        net.paramRef();
        
        // 4.  Execution parameters definition and start computing
        int epochs = 100;                                             // How many times we treat the entire training data set
        boolean shuffle = true;                                       // Shuffle or not training sets between epochs
        net.start(epochs, shuffle);
    }
}
```

A screenshot from an execution with early achievement of 98s %:

![Alt text](https://github.com/aisasa/DLNetworkJ/blob/master/A%20promising%20start.png "A promising start!")

Educational purposes only, without warranty of any kind (see license).
