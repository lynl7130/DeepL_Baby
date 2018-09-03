
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
  grad_square += dx * dx #Added element-wise scaling of the gradient based on the historical sum of squares in each dimension
  x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```
* What if high condition number?  
if dw1 is small, grad_square's dimension 1 will be small too => moving faster in this dimension.  
* Adagrad's feature?  
over long time, the step size is getting smaller and smaller.  
1. Good in convex case: want the step size getting smaller to reach the minima.  
2. Bad in non-convex case: might get stuck at saddle point.  
  
#### RMSProp: solving Adagrad's problem


## Regularization
  
## Transfer Learning
  
