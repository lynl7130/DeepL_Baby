
# Loss Functions and Optimization

Recall from last time: linear classifier
## loss function 
tells how good our current classifier is.  
Given a dataset of examples, loss:  
![](https://latex.codecogs.com/gif.latex?L%20%3D%20%5Cfrac%7B1%7D%7BN%7D%5Csum_i%7BL_i%20%28f%28x_i%2C%20W%29%2C%20y_i%29%7D)
**score function:** ![](https://latex.codecogs.com/gif.latex?s%20%3D%20f%28x_i%2C%20W%29)   
  
### Hinge loss(Multi-class SVM loss):  
![](https://latex.codecogs.com/gif.latex?L_i%20%3D%20%5Csum_%7Bj%5Cneq%20%7By_i%7D%7D%7Bmax%280%2C%20s_j%20-%20s_%7By_i%7D%20&plus;%201%29%7D)   
  target: make all the scores lower than the target class score.  
  finally, ![](https://latex.codecogs.com/gif.latex?L%3D%20%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7DL_i)  
    
  **Questions:**  
  1. **1: the margin**  
  2. Minimum loss: 0; maximum loss: infinity.  
  3. If W is so small that all s ~ 0, the loss will be C-1.(could be used in debugging!)  
  4. Why let out the correct class? make the minimum loss = 0(human interpretation).  
  5. What if we used mean instead of sum? Doesn't matter! **We do not care true value of loss**.   
  6. What if we used ![](https://latex.codecogs.com/gif.latex?L_i%20%3D%20%5Csum_%7Bj%5Cneq%7By_i%7D%7Dmax%280%2C%20s_j%20-%20s_%7By_i%7D%20&plus;%201%29%5E2)?  matters!  
  **square<->linear: square hate big error, while linear treat big/small error identically** 
  7. Suppose that we found a W such that L=0, is this W unique? NO! **rescale**!
  
### Regularization  
![](https://latex.codecogs.com/gif.latex?L%28W%29%20%3D%20%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5ENL_i%28f%28x_i%2C%20W%29%2Cy_i%29%20&plus;%20%5Clambda%20R%28W%29) 
**Data Loss**: Model predictions should match training data.  
We don't care about performance on training data, instead we care about performance on unseen data(test data).  
**Regularization**: Model should be "simple", so it works on test data.  
**Occam's Razor**: Among competing hypotheses, the simplest is the best.
* In common use:  
  1. L2 regularization(Weight Decay): ![](https://latex.codecogs.com/gif.latex?R%28W%29%20%3D%20%5Csum_k%5Csum_l%20W_%7Bk%2Cl%7D%5E2)  
  2. L1 regularization: ![](https://latex.codecogs.com/gif.latex?R%28W%29%20%3D%20%5Csum_k%5Csum_l%7CW_%7Bk%2Cl%7D%7C)  
  3. Elastic net(L1+L2): ![](https://latex.codecogs.com/gif.latex?R%28W%29%20%3D%20%5Csum_k%5Csum_l%5Cbeta%20W_%7Bk%2Cl%7D%5E2&plus;%7CW_%7Bk%2Cl%7D%7C)  
  4. Max norm regularization.  
  5. Dropout.  
  6. Fancier: Batch normalization, stochastic depth.  
  
### Softmax Classifier (Multinomial Logistic Regression)
Scores = unnormalized log probabilities of the classes.  
![](https://latex.codecogs.com/gif.latex?P%28Y%3Dk%7CX%3Dx_i%29%20%3D%20%5Cfrac%7Be%5E%7Bs_k%7D%7D%7B%5Csum_je%5E%7Bs_j%7D%7D)  
maximize the log likelihood of the correct class -> minimize the negative log likelihood of the correct class:  
![](https://latex.codecogs.com/gif.latex?L_i%20%3D%20-logP%28Y%3Dy_i%7CX%3Dx_i%29)  
* Why log?  
  We want the probability to be close to 1.  
  With log: loss->0 means prob->1.  
  
**Questions:**   
  1. minimum loss:0; maximum loss: infinity.  
  2. What if S is too small? ![](https://latex.codecogs.com/gif.latex?logC)  
 
**Softmax vs. SVM:**  
Softmax: probability distribution, want the probability to predict correct class -> 1.  
         Always want the score to be better and better.  
SVM: calculate margin, want the score of incorrect class to be far lower than those of correct class.  
     If the score is far away enough, won't care.  
     
## Optimization

### Strategy #1: Random Search
Don't use in practice!  

### Staregty #2: Follow the slope
What is slope?
* In 1-dimension, the derivative of a function.  
* In multiple dimensions, the **gradient** is the vector of (partial derivatives) along each dimension.  
The slope in any direction is the dot product of the direction with the gradient.  
The direction of steepest descent is the **negative gradient**.  
    
Use calculus to compute on **analytic gradient**!  
**gradient check**: check implementation with **numerical gradient**(slow, so only for check)!
  
* Gradient points to the increasing direction of loss function!

### Gradient Descent
```
# Vanilla Gradient Descent

# Initialized weights as some value
while True:
  weights_grad = evaluate_gradient(loss_fun, data, weights)
  weights += - step_size * weights_grad # perform parameter update
  
```
step_size(learning rate): a hyperparameter. **The first thing to set!**

### Stochastic Gradient Descent(SGD)
Why SGD?  
* Full sum expensive when N is large!  
* Approximate sum using a **minibatch** of examples. (32/64/128 common)   
```
# Vanilla Minibatch Gradient Descent

while True:
  data_batch = sample_training_data(data, 256) # sample 256 examples
  weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
  weights += - step_size * weights_grad # perform parameter update

```

### Image Features
Motivation: after applying feature transform, points can be separated by linear classifier.  
**Examples:**   
  * Color Histogram
  * Histogram of Oriented Gradients(HoG)
  * Bag of Words
