
# Training Neural Networks I

## One time setup
### Activation Functions
activation function: provides non-linearity.  
#### Sigmoid 
* Squashes numbers to range:  ```[0,1]```  
* Historically popular since they have nice interpretation as a saturating "firing rate" of a neuron.  
**3 problems:**   
  1. Saturated neurons "kill" the gradients.  
  2. Sigmoid outputs are not zero-centered:  
  **We don't want dw1, dw2, dw3... to be all positive nor all negative. -> w move like zigzag.**  
  3. exp() is a bit compute expensive.  
#### Tanh
* Squashes number to range: ```[-1,1]```  
* zero centered(**nice**)  
* still kills gradients when saturated.  
#### ReLU(Rectified Linear Unit)
* Does not saturate(in + region)
* Very computationally efficient
* Converges much faster than sigmoid/tanh in practice.(e.g. 6 times faster)
* Actually more biologically plausible than sigmoid.  
**2 problems:**
  1. Not zero-centered output
  2. An annoyance: what is the gradient when x<0?   
dead ReLU: will never activate => never update.(a bad initialization? too big learning rate?)  
**=> people like to initialize ReLU neurons with slightly positive biases(e.g. 0.01)**  
#### Leaky ReLU
![](https://latex.codecogs.com/gif.latex?f%28x%29%3Dmax%280.01x%2C%20x%29)  
* Does not saturate
* Computationally efficient
* Converges much faster than sigmoid/tanh in practice.(e.g. 6 times faster)
* **will not die!**
#### Parametric Rectifier(PReLU)
![](https://latex.codecogs.com/gif.latex?f%28x%29%3Dmax%28%5Calpha%20x%2C%20x%29)  
backprop and learn the alpha!  
#### Exponential Linear Units(ELU)
![](https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20x%20%26%20x%20%3E%200%5C%5C%20%5Calpha%28exp%28x%29-1%29%20%26%20x%5Cleq%200%20%5Cend%7Bmatrix%7D%5Cright.)  
* All benefits of ReLU
* Closer to zero mean outputs
* Negative saturation regime compared with Leaky ReLU adds some robustness to noise.  
**problem:** Computation requires exp().  
#### Maxout "Neuron"
![](https://latex.codecogs.com/gif.latex?max%28w_1%5ETx%20&plus;%20b_1%2C%20w_2%5ETx&plus;%20b_2%29)  
* Does not have the basic form of dot product -> nonlinearity
* Generalizes ReLU and Leaky ReLU
* Linear Regime! Does not saturate! Does not die!  
**problem:** doubles the number of parameters/neuron.  
  
#### TLDR: In practice:
* Use ReLU. Be careful with your learning rates
* Try out Leaky ReLU/Maxout/ELU
* Try out tanh but don't expect too much
* Don't use sigmoid

### Data Preprocessing
Step 1: preprocess the data.  
original -> zero-centered.  
More complicated methods: normalize? PCA? Whitening? => too complicated for image!  
**Why need normalization?**  
Before: classification loss very sensitive to changes in weight, hard to optimize.  
After: less sensitive to small changes in weights, easier to optimize.  
  
#### TLDR: In practice for Images: center only
e.g. consider CIFAR-10 example with (32,32,3) images  
* Subtract the mean image(e.g. AlexNet), a (32,32,3) array
* Subtract per-channel mean(e.g. VGGNet), mean along each channel = 3 numbers.  
**Mean does not come from single batches! Want the experience from entire training dataset.**  


### Weight Initialization
Q: What if W=0 is used? A: all neurons will do the same thing(same gradient, same output...).
#### Small random numbers
(gaussian with zero-mean and 0.01 standard deviation)  
```W = 0.01 * np.random.randn(fan_in, fan_out)```  
**Problem:**  
Works fine for small networks, but **bad for deeper networks**:  
Later activations become zero => **gradients too small** to update.  
#### Large random numbers
(gaussian with zero-mean and 1 standard deviation)    
```W = 1 * np.random.randn(fan_in, fan_out)```   
**Problem:**  
Almost all neurons completely saturated, either -1 and 1. => **Gradients wil be all zero.**  
#### Xavier initialization: reasonable!
standard deviation too small => collapse, too big => saturate.  
```W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)```  
**Point**: want the variance of the input to be the same as the variance of the output.  
i.e. If there's a layer with few neurons, want the weight to be bigger to transport enough output to next layer.  
  
**Assumption**: linear activation, e.g. in the active region of tanh.  
**Problem:** When using ReLU nonlinearity it breaks, because ReLU approximately kill half of the neurons => weights too small.  
**Solution**: ```W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in/2)```  

### Batch Normalization: make unit gaussian activations
consider a batch of activations at some layer. To make each dimension unit gaussian:  
![](https://latex.codecogs.com/gif.latex?%5Cwidehat%7Bx%7D%5E%7B%28k%29%7D%20%3D%20%5Cfrac%7Bx%5E%7B%28k%29%7D-E%5Bx%5E%7B%28k%29%7D%5D%7D%7B%5Csqrt%7BVar%5Bx%5E%7B%28k%29%7D%5D%7D%7D)  
![](https://latex.codecogs.com/gif.latex?E%5Bx%5E%7B%28k%29%7D%5D): mean of the current batch.  
![](https://latex.codecogs.com/gif.latex?%5Csqrt%7BVar%5Bx%5E%7B%28k%29%7D%5D%7D): standard deviation of the current batch. 
**Steps:**  
1. compute the empirical mean and variance independently for each dimension.  
2. Normalize(apply the formula).  
  
**Where?**  
Usually inserted after Fully Connected or Convolutional layers, and before nonlinearity.  
* for Convolutional layers, normalize not just across the training examples, but also across feature dimension and spatial locations.  
  
**Problem:** do we necessarily want a unit gaussian input to a tanh layer?  
**Solution:** After the normalization, allow the network to squash the range if it wants to:  
![](https://latex.codecogs.com/gif.latex?y%5E%7B%28k%29%7D%20%3D%20%5Cgamma%5E%7B%28k%29%7D%5Cwidehat%7Bx%7D%5E%7B%28k%29%7D&plus;%5Cbeta%5E%7B%28k%29%7D)  
* The network could learn to recover the identity mapping:  
![](https://latex.codecogs.com/gif.latex?%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20%5Cgamma%5E%7B%28k%29%7D%3D%5Csqrt%7BVar%5Bx%5E%7B%28k%29%7D%5D%7D%5C%5C%20%5Cbeta%5E%7B%28k%29%7D%3DE%5Bx%5E%7B%28k%29%7D%5D%20%5Cend%7Bmatrix%7D%5Cright.)  
  
**BN's benefits:**  
  1. Improves gradient flow through the network.  
  2. Allows higher learning rates.  
  3. Reduces the strong dependence on initialization.
  4. Acts as a form of regularization in a funny way(using experience from other data in that minibatch), and slightly reduces the need for dropout, maybe
  
**At test time BN layer functions differently!**
Test time: Then mean/std are computed based on the batch(e.g. can be estimated during traning with running averages)



## Training dynamics

### Babysitting the Learning Process
#### Step 1: PreProcess the data.  
#### Step 2: Choose the architecture.  
#### Step 3: Double check(sanity) that the loss is reasonable:  
1. Do a forward pass, **disable regularization**, make sure the implemenation is right  
2. Do a forward pass, **crank up regularization**, want to see the loss goes up  
#### Step 4: Let's try to train now...  
Make sure that you can overfit very small portion of the training data!  
1. take a very small portion of data.  
2. turns off regularization.  
3. train and see the loss, want to see to loss drop down to zero.  
#### Step 5: Let's really try to train:
Start with small regularization and find learning rate that makes the loss go down.  
* loss barely changes => learning rate is too low.  
* loss exploding => learning rate is too high.  
  
**rough learning rate range:** ```(0.00001, 0.001)```


### Hyperparameter Optimization
#### Cross-validation strategy
**coarse -> fine** cross-validation in stages  
1. **First stage**: only a few epochs to get rough idea of what params work  
* it's best to optimize in log space, e.g. ```lr = 10 ** uniform(-3, 6) ```, because the learning rate is multiplying gradient update, so it has these multiplicative effects, so it makes more sense than ```lr = uniform(10^-6, 10^-3)```.  
  
2. **Second stage**: longer running time, finer search  
* Make sure you've fully explored your range!  
  
3. **...(repeat as necessary)**  
  
* Tip for detecting explosions in the solver:  
If the cost is ever > 3 * original cost, break out early.   

#### Sample all of hyperparameters: Random Search vs. Grid Search
* Random Search is better:  
Easier to observe the shape of hyperparameter groups: where the good values are.  
  
#### Hyperparameters to play with:
* network architecture  
* learning rate, its decay schedule, update type  
* regularization(L2/Dropout strength)

## Evaluation

### model ensembles

#### Monitor and visualize the loss curve:
* loss exploding => very high learning rate
* loss going down slowly => low learning rate
* loss going down quickly but reach a plateau => high learning rate
* loss going down quickly => good learning rate

#### Why loss flat for a while then start training at a sudden?
* a prime suspect: bad initialization

#### Monitor and visualize the accuracy:
compare: training accuracy <=> validation accuracy  
* big gap = overfitting => increase regularization strength?  
* no gap => increase model capacity? Because has not overfitted yet, could increase more.  

#### Track the ratio of weight updates/weight magnitudes:
ratio between the udpates and values: ~ 0.0002 / 0.02 = 0.01 (about okay)  
**want this to be somewhere around 0.001 or so**
