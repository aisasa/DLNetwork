# A Deep Learning Network with Java

A basic Java deep learning network to attack the MNIST data set of hand-written digits, but easily customizable for other purposes. 
Implements net shape configuration, quadratic and cross-entropy cost functions, sigmoid function in neurons, basic (lineal) adaptative 
learning rate, mini-batch options... As rules of the game are clearer when playing with understandable figures instead of bytes, a 
zipped text version of the MNIST data sets (training, validation and test) is also provided, besides Java arrays sets as zipped-saved files.

The project is based on the first two chapters of Michael Nielsen's work "Neural networks and deep learning", published on 
https://neuralnetworksanddeeplearning.com and https://github.com/mnielsen/neural-networks-and-deep-learning. My acknowledgment and 
gratitude to the author for such a magnificent job.

This project is not a direct translation between Python (as in the Michael's work) and Java, and keeps its own particularities 
both in the structure of the files and in the program's flow. It has a few more lines of core code because of Java strongly typed
nature as well as some other causes. For instance, no external numeric library has been applied, so the necessary mathematical 
functions have been implemented in their basic incarnations in the corresponding file (DLMath.java). Don't expect, therefore, an optimized 
performance. Also it is not a good example of object oriented programming due their design fundamentals, adhered to the more algebraic 
Michael's approach instead to consider neuron abstractions.

Building this software has been one of the best ways to follow and understand the fundamentals of deep learning that Michael Nielsen 
explains in his work. 

The software works, and in particular conditions (cross-entropy cost function, with mini-batch=1 or with minimum size, learn rate = 0.07, 
initialization of weights with Math.random () / 100 and biases with Math.random () / 50) is able to obtain, in a moderate amount of epochs 
(say 20-40), between 98.30% and 98.40% of success in the recognition of the MNIST set, with scores in 98.50%-98.60% in less than a hundred
epochs.

Use: ready to load data sets in zipped files. In the source directory execute DLRun.java as usual. Modify parameters (in DLRun.java) and 
code (surely in DLNetwork.java, main(), DLInit.java, or whatever) as needed. 

Educational purposes only, without warranty of any kind (see license).
