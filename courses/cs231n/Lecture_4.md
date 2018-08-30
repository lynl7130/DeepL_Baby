
# Backpropagation and Neural Networks
  
## Backpropagation
  
### Computational graphs: for Backpropagation . 
Backpropagation: make use of local gradient.  
  
### sigmoid function:  
![](https://latex.codecogs.com/gif.latex?%5Csigma%20%28x%29%3D%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-x%7D%7D)  
sigmoid gate:  
![](https://latex.codecogs.com/gif.latex?%5Cfrac%7Bd%5Csigma%20%28x%29%7D%7Bdx%7D%3D%281-%5Csigma%28x%29%29%5Csigma%28x%29)  
  
### Patterns in backward flow
* add gate: gradient distributor.  
* max gate: gradient router.  
* mul gate: gradient switcher.  
  
### Gradients for vectorized code
x, y, z are now vectors.  
Gradient is now the **Jacobian matrix**(derivative of each element of z w.r.t. each element of x)  
  
### Modularized implementation
forward/backward API:  
```
class MutiplyGate(object):
  def forward(x, y):
    z = x * y
    self.x = x
    self.y = y
    return z
  
  def backward(dz):
    dx = self.y * dz
    dy = self.x * dz
    return [dx, dy]

```
  
## Neural Networks
  
Neural Networks: stack multiple linear functions together with non-linear functions in between.  
* Be very careful with your brain analogies!  

### Activation functions
* Sigmoid
* tanh
* ReLU
* Leaky ReLU
* Maxout
* ELU
...

### Architectures
* 2-layer Neural Net(1-hidden-layer Neural Net)
* 3-layer Neural Net(2-hidden-layer Neural Net)
...
