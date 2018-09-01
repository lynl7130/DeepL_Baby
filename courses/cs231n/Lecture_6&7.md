
# Training Neural Networks

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
  
### TLDR: In practice:
* Use ReLU. Be careful with your learning rates
* Try out Leaky ReLU/Maxout/ELU
* Try out tanh but don't expect too much
* Don't use sigmoid

### Data Preprocessing
### Weight Initialization
### Batch Normalization
### Babysitting the Learning Process
### Hyperparameter Optimization

## Training dynamics

## Evaluation
