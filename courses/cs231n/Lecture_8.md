
# Deep Learning Software


## CPU vs. GPU
**CPU**:  
* fewer cores, but each core is much faster and much more capable.   
* great at sequential tasks.  
* cache is relative small, using system RAMs(12/16/32G).  
  
**GPU**:  
* more cores, but each core is much slower and "dumber".  
* great for parallel tasks.  
* have own cache and own RAM built in the chip, given it's slow to comunicate between system RAM and GPU.   

### NVDIA vs. AMD
NVDIA wins!

### Example: Matrix Multiplication -> something suitable for GPUs!
```
    B            C             C
A       x    B        =    A
```
imagine having AxC cores doing the same thing:  
read in a row and a column then output a number!  => super fast with GPU.  

### Programming GPUs
#### CUDA(NVDIA only)
* Write C-like code that runs directly on the GPU  
complicated! => use APIs:  
* Higher-level APIs: cuBLAS, cuFFT, cuDNN, etc.  
#### OpenCL  
* similar to CUDA, but runs on anything: NVDIA GPUs, AMDs, CPUs  
* Usually slower: has not yet been optimized  
  
#### Learn to write!
* Udacity: Intro to Parallel Programming  

## Deep Learning Frameworks
from academy to industry
* Caffe -> Caffe2
* Theano -> Tensorflow
* Torch -> PyTorch

### Why need deep learning frameworks?
* Easily build big computational graphs.  
* Easily compute gradients in computational graphs.  
* Run it all efficiently on GPU.  
**Numpy is CPU only! + cannot automatically compute gradient!**  
  
### Tensorflow  
#### First Define computational graph:  
1. create placeholders for inputs, weights and targets.  
2. Build forward pass: compute prediction for target and loss.(**No computation here!**)  
3. Design gradient(**No computation here!)  
#### Then run the graph many times:  
1. enter a Tensorflow session to run the graph.  
2. create dictionary of numpy arrays to fill in the placeholders.  
3. run the graph; get numpy arrays for loss, grad_w1 and grad_w2.  
#### Train the network:  
Run the graph over and over, use gradient to update weights.  
  
**Problem: copying weights between CPU and GPU: slow.**  
**Solve:**   
1. store weights in tf.Variables instead of placeholder -> **Weights now live inside the graph!**.  
2. use w1.assign function to initialize weights.(**No computation!**)  
3. Run graph once to initialize w1 and w2.  
4. Run many times to train.  
  
**Problem: loss is not moving!**  
**Reason:**  
Tensorflow only executes steps that are necessary for computing the targets, thus is not updating the weights!  
**Add dummy graph node that depends on updates?**  
The output of tf.group is not tf.Variable -> copying between CPU and GPU!  
-> tf.group!  
  
**Question: why put inputs in placeholder instead of Variable?**  
Inputs could be different in every iteration.  
  
#### TensorFlow: Optimizer
**Solve the loss not moving error!**  
Can use an optimizer to compute gradients and update weights.  
* Remember to execute the output of the optimizer!  
  
#### TensorFlow: Loss
Use predefined common losses.  
  
#### TensorFlow: Layers
1. Use Xavier initializer.  
2. tf.layers automatically sets up weight(and bias) for us!  
  
#### Keras: High-Level  
Keras: a layer on top of TensorFlow, makes common things easy to do.  
(Also supports Theano backend)  
1. Define model object as a sequence of layers.  
2. Define optimizer object.  
3. Build the model, specify loss function.  
4. Train the model with a single line.
**There are other high-level wrappers!**  

#### TensorFlow: Pretrained Models
  
#### TensorFlow: Tensorboard  
  
#### TensorFlow: Distributed Version  
Split one graph over multiple machines!  

#### Side Node: Theano
TensorFlow is similar in many ways to Theano(earlier framework from Montreal).  

### PyTorch: Three Levels of Abstraction
**Have equivalents in TensorFlow!**  
**Tensor**: Imperative ndarray, but runs on GPU.  
**Variable**: Node in a computational graph; stores data and gradient.  
**Module**: A neural network layer; may store state or learnable weights.  
  
#### PyTorch: Tensors
~numpy arrays but can run on GPU.  

