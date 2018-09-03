
# Training Neural Networks II
  
## Fancier optimization
### Problems with SGD
**1. What if loss changes quickly in one direction(say,w1) and slowly in another(say, w2)?**  
* The loss is very sensitive to the changes in w1.  
* Loss function has high **condition number**: ratio of largest to smallest singular value of the Hessian matrix is large.  
* very slow progress along shallow dimension, jitter along steep direction.  
       <- because the direction of gradient does not align with the direction towards minima!  
* more params, more serious the problem become    
  
**2. What if the loss function has a local minima or saddle point**?  
* Zero gradient, gradient descent gets stuck.  
* larger network, more saddle problem than local minima problem.
  
**3. Our gradients come from minibatches so they can be noisy!**

### A technique to solve all these problems: SGD + Momentum
```
#SGD
while True:
  dx = compute_gradient(x)
  x -= learning_rate * dx
  
#SGD + Momentum
vx = 0 #velocity: a running mean of gradients
while True:
  dx = compute_gradient(x)
  vx = rho * vx + dx #rho: friction, typically=0.9/0.99
  x -= learning_rate * vx  
```
#### Nesterov Momentum: correct Momentum with current velocity and previous velocity  
vx= incorporate vx and the gradient at the point you will reach by simply following the raw vx.  
![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bmatrix%7D%20v_%7Bt&plus;1%7D%20%3D%20%5Crho%20v_t%20-%20%5Calpha%20%5CDelta%20f%28x_t&plus;%5Crho%20v_t%29%5C%5C%20x_%7Bt&plus;1%7D%20%3D%20x_t%20&plus;%20v_%7Bt&plus;1%7D%20%5Cend%7Bmatrix%7D)  
Annoying, usually we want update in terms of x and delta. So:  
![](https://latex.codecogs.com/gif.latex?%5Cwidetilde%7Bx_t%7D%20%3D%20x_t%20&plus;%20%5Crho%20v_t)  
```
v = 0
while True:
  dx = compute_gradient(x) # change: x = x + rho * v
  old_v = v
  v = rho * v - learning_rate * dx
  x += -rho * old_v + (1 + rho) * v 

```
* Nesterov Momentum over Momentum?  
velocity correction factor. Help alleviate overshooting.   
  
### Another technique: AdaGrad
```
grad_squared = 0
while True:
  dx = compute_gradient(x)
  grad_squared += dx * dx #Added element-wise scaling of the gradient based on the historical sum of squares in each dimension
  x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7) #1e-7: make sure not dividing 0
```
* What if high condition number?  
if dw1 is small, grad_square's dimension 1 will be small too => moving faster in this dimension.  
* Adagrad's feature?  
over long time, the step size is getting smaller and smaller:  
  1. Good in convex case: want the step size getting smaller to reach the minima.  
  2. Bad in non-convex case: might get stuck at saddle point.  
  
#### RMSProp: solving Adagrad's problem
Momemtum over the squared gradients, rather than momentum over the gradient.  
```
grad_squared = 0
while True:
  dx = compute_gradient(x)
  grad_squared += decay_rate * grad_squared + (1 - decay_rate) * dx * dx
  x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```

#### SGD vs. SGD+Momentum vs. Adagrad vs. RMSProp
* Both SGD+Momentum and RMSProp work better than SGD.  
* SGD+Momentum tends to overshoot the miminum and comes back.  
* RMSProp adjusts its direction such that it has equal speed at each dimension.  
* Adagrad looks kind of RMSProp, but tends to stuck.  

### Adam
  
#### combining Momentum & AdaGrad/RMSProp: Almost there!
```
first_moment = 0
second_moment = 0
while True:
  dx = compute_gradient(x)
  first_moment = beta1 * first_moment + (1 - beta1) * dx   #Momentum
  second_moment = beta2 * second_moment + (1 -beta2) * dx * dx  #AdaGrad/RMSProp  
  x -= learning_rate * first_moment / (np.sqrt(second_moment) + 1e-7))
```
* What happens at the first timestep?  
second_moment is still very close to zero(typically, beta2 = 0.9/0.99) => first step is large and has nothing to do with problem geometry  
  
#### Adam(full form)
* Why Bias correction?  
For the fact that first and second moment estimates start at zero.  
```
first_moment = 0
second_moment = 0
for t in range(num_iterations):
  dx = compute_gradient(x)
  first_moment = beta1 * first_moment + (1 - beta1) * dx   #Momentum
  second_moment = beta2 * second_moment + (1 -beta2) * dx * dx  #AdaGrad/RMSProp  
  first_unbias = first_moment / (1 - beta1 ** t)
  second_unbias = second_moment / (1 - beta2 ** t)
  x -= learning_rate * first_unbias / (np.sqrt(second_unbias) + 1e-7))
```
**Adam with beta1 = 0.9, beta2 = 0.999, learing_rate = 1e-3 or 5e-4 is a great starting point for many models!**
  
#### Adam vs. SGD/SGD+Momentum/RMSprop
Adam combines elements of SGD+Momentum and AdaGrad/RMSProp.  
* Adam overshoots the minumum a little bit like SGD+Momentum, but it does not overshoot that much.  
* Adam tries to walk at equal speed in all dimensions.  

### Learning rate
SGD, SGD+Momentum, Adagrad, RMSProp, Adam all have learning rate as a hyperparameter.  

#### Learning rate could decay over time!
**Several decay types:**  
* step decay: e.g. decay learning rate by half every few epochs.  
* exponential decay:  
![](https://latex.codecogs.com/gif.latex?%5Calpha%20%3D%20%5Calpha_0%20e%5E%7B-kt%7D)  
* 1/t decay:  
![](https://latex.codecogs.com/gif.latex?%5Calpha%20%3D%20%5Calpha_0/%20%281&plus;kt%29)  
  
**Why Loss: drop -> flat -> drop -> flat... ?**  
Learning rate decay happens!  
Getting near the minima, start bouncing around -> learning rate smaller -> could get closer and closer.  
  
**When to use learning rate decay?**  
Common in SGD+momentum, a little bit less common in Adam.  
  
**Learning rate decay is a second-order hyperparameter!**  
At the start of training, pick a good learning rate with no learning rate decay instead.  
  
### First/Second-Order Optimization
  
#### First-Order Optimization
1. Use gradient form linear approximation.  
2. Step to minimize the approximation.  
#### Second_Order Optimization: nice because no learning rate!  
1. Use gradient and Hessian to horm quadratic approximation.  
2. Step to the minima of the approximation.  
solving for the critical point we obtain the Newton parameter update:  
![](https://latex.codecogs.com/gif.latex?%5Ctheta%5E*%20%3D%20%5Ctheta_0%20-%20H%5E%7B-1%7D%5CDelta_%7B%5Ctheta%7DJ%28%5Ctheta_0%29)  
**Impractical for deep learning!**  
Hessian has O(N^2) elements. Inveriting takes O(N^3).  
**How to solve this problem?**  
1. Quasi-Newton methods(**BGFS** most popular):  
instead of inverting the Hessian(O(N^3)), approximate inverse Hessian with rank 1 updates over time (O(N^2) each).  
2. **L-BFGS**(Limited memory BFGS):  
Does not form/store the full inverse Hessian.  
* Usually works very well in full batch, deterministic mode.  
* Does not transfer very well to mini-batch setting.  

#### In practice:
* **Adam** is a good default choice in most cases(better for NNs).  
* If you can afford to do full batch updates then try out **L-BFGS**(and don't forget to disable all sources of noise).    
  
### Beyond Training Error  
We really care about error on new data - how to reduce the gap?   

#### Model Ensembles: Enjoy 2% extra performace
1. Train multiple independent models.  
2. At test time average their results.  

#### Model Ensembles: Tips and Tricks
* Instead of training independent models, use multiple snapshots of a single model during training!  
snapshot: responding to a minima.  
* Instead of using actual parameter vector, keep a moving average of the parameter vector and use that at test time(Polyak averaging).   

## Regularization: improve single-model performance
  
### Dropout
In each forward pass, randomly set some neurons to 0.  
* Probability of dropping is a hyperparameter; 0.5 is common.   
```
#Example forward pass with a 3-layer network using dropout
p = 0.5 # probability of keeping a unit active. higher = less dropout

def train_step(X):
  H1 = np.maximum(0, np.dot(W1, X) + b1)
  U1 = np.random.rand(*H1.shape) < p # first dropout mask
  H1 *= U1 # drop!
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  U2 = np.random.rand(*H2.shape) < p # second dropout mask
  H2 *= U2 # drop!
  out = np.dot(W3, H2) + b3
  
  # backward pass: compute gradients ...
  # perform parameter update...
  
```  
#### How could Dropout be a good idea?  
* Forces the network to have a redundant representation; Prevents co-adaptation of features.  
* Dropout is training a large ensemble of models(that share parameters).  

#### Dropout: Test time
Dropout makes our output random!  
Want to "average out" the randomness at test-time:  
![](https://latex.codecogs.com/gif.latex?y%20%3D%20f%28x%29%20%3D%20E_z%5Bf%28x%2Cz%29%5D%20%3D%20%5Cint%20p%28z%29%20f%28x%2Cz%29dz)  
z: random mask.  
But this integral seems hard... Want to approximate the integral:  
```
# At test time all neurons are active always
# => we must scale the activations so that for each neuron:
#    output at test time = expected output at training time 
def predict(X):
  #ensemble forward pass
  H1 = np.maximum(0, np.dot(W1, X) + b1) * p # scale the activation
  H2 = np.maximum(0, np.dot(W2, H1) + b2) * p # scale the activation
  out = np.dot(W3, H2) + b3
```

#### More common: "Inverted dropout"
```
p = 0.5

def train_step(X):
  H1 = np.maximum(0, np.dot(W1, X) + b1)
  U1 = (np.random.rand(*H1.shape) < p) / p # first dropout mask, Notice /p!
  H1 *= U1
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  U2 = (np.random.rand(*H2.shape) < p) / p # second dropout mask, Notice /p!
  H2 *= U2
  out = np.dot(W3, H2) + b3
  
  # backward pass: compute gradients ...
  # perform parameter update...

def predict(X):
  #ensemble forward pass
  H1 = np.maximum(0, np.dot(W1, X) + b1) # no scaling necessary
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  out = np.dot(W3, H2) + b3

```
  
* It usually takes longer to train dropout, because each time you are training a subpart of this network. But the trained model tends to generalize better.   

### Regularization: A common path
**Training**: Add some kind of randomness: 
![](https://latex.codecogs.com/gif.latex?y%20%3D%20f_W%28x%2Cz%29)  
**Testing**: Average out randomness(Sometimes approximate):  
![](https://latex.codecogs.com/gif.latex?y%20%3D%20f%28x%29%20%3D%20E_z%5Bf%28x%2Cz%29%5D%20%3D%20%5Cint%20p%28z%29%20f%28x%2Cz%29dz)  

#### Example: Dropout

#### Example: Batch Normalization
Training: Normalize using stats from random minibatches.  
Testing: Use fixed stats to normalize.  
#### Example: Data Augmentation
randomly transform the image:
* Horizontal Flips
* Random crops and scales
* Color Jitter
...
  
#### Example: DropConnect
#### Example: Fractional Max Pooling
#### Example: Stochastic Depth

#### In practice
* Have Batch Normalization, because it helps you converge, tends to be enough.  
* Once you see your network overfitting, try adding other things.  

## Transfer Learning
Only have a small dataset, but there's a large dataset available that helps give features you need.  

### Steps:
1. Train on a large dataset(ImageNet).  
2. fine-tuning on your own dataset with lower learning-rate(1/10 of original LR is a good start!):  
* Small Dataset(C classes):  
Freeze all the other layers. Reinitialize the last fully-connected layer and train only on this layer to fit the C classes small dataset.  
* Bigger Dataset:  
train more fc layers.  

### When and how?
```
                  |    very similar dataset  |  very different dataset
-------------------------------------------------------------------------
very little data  |   Use Linear Classifier  |   Try linear classifier 
                  |        on top layer      |   from different stages
-------------------------------------------------------------------------                           
quite a lot data  |   Finetune a few layers  |  Finetune a larger number                        
                  |                          |          of layers
-------------------------------------------------------------------------
top: more specific
bottom: more generic
```
### Transfer learning with CNNs is pervasive!  
It's the norm, not an exception.  
Don't start from scratch! start from pretrained ones!  
  
### Takeaway for your projects and beyond:  
Have some dataset of interest but it has < ~1M images?  
1. Find a very large dataset that has similar data, train a big ConvNet there.  
2. Transfer learn to your dataset.  
  
* Deep learning frameworks provide a "Model Zoo" of pretrained models so you don't need to train your own.  
