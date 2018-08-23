
# Image Classification

### Semantic Gap
**semantic**: car is the semantic label assigned to an image.  
**gap**: semantic idea of an image <-> the numbers the computer sees.  

### Challenges
  1. Viewpoint Variation.  
  2. Illumination.  
  3. Deformation.  
  4. Occlusion.  
  ...  

### Solve: Data-Driven Approach

Collect Data -> Get Classifier -> Evaluate Classifier.  

Image Classfication APIs:
```
Model train(input_images, labels)
Labels predict(model, test_images)
```

### First classifier: Nearest Neighbor
**train**: Memorize all data and labels.
**predict**: predict the label of the most similar training image. 
Example Dataset: CIFAR10  
* What does similar mean?  
   **Distance Metric** to compare images:  
   1. L1 distance(Manhattan distance):  
   ![](https://latex.codecogs.com/gif.latex?d_1%28I_1%2C%20I_2%29%20%3D%20%5Csum_p%20%7C%7BI_1%5Ep-I_2%5Ep%7D%7C)  
   2. L2 distance(Euclidean distance):  
   ![](https://latex.codecogs.com/gif.latex?d_2%28I_1%2C%20I_2%29%20%3D%20%5Csqrt%7B%5Csum_p%20%28%7BI_1%5Ep-I_2%5Ep%7D%29%5E2%7D)
   ...  
* How fast are training and prediction?  
  Train: O(1), predict: O(N).  
  **Bad!** Want classifiers that are fast at prediction; slow for training is ok.  
* What does this look like?  
  split into colored regions!  
  Problem: noise points/ overfitting => **K-Nearest Neighbors**  

#### K-Nearest Neighbors
Instead of copying label from nearest neighbor, take majority vote from K closest points
K up => removing noise & boundary becoming smoothy

**KNN is a general algorithm!*** Just choose K & choose suitable distance metrics.   

#### KNN is never used on images
* Very slow at test time  
* Distance metrics on pixels are not informative  
* curse of dimensionality  

### Hyperparameters
choices about the algorithm that we set rather than learn. (e.g. K, distance metrics)    
#### try and find the best!  
Idea #1: Choose hyperparameters that work best on the data.  
  * BAD! K=1 always works perfectly on training data  
Idea #2: Split data into train and test, choose hyperparameters that work best on test data.  
  * BAD! No idea how algorithm will perform on new data  
Idea #3: Split data into train, val and test, choose hyperparameters on val and evaluate on test.   
  * BETTER! the performance on testing set is telling you how good your algorithm is doing on unseen data.  
Idea #4: Split data into folds and a test, try each fold as validation and average the results.      
   e.g. 5-fold: train on 4 folds, evaluate on 1, switch 5 times, then evaluate on test  
   Useful for small datasets, but not used too frequently in deep learning  
**test dataset might not be representative of the general distribution, but could alleviate by dataset creators.**   

### Linear Classification
NNs are like Legos, and linear classifiers are among the most basic component.  

#### Parametric Approach
```
 Image -> f(x, W) -> numbers giving class scores(larger score-> larger possibility)
```
* W: parameters or weights  
**Linear Classifier**: f(x, W) = Wx + b  
* b: bias, e.g. many cats in this dataset, than add more to cat class  
**Interpreting a linear classifier**:  
1. linear classifier is only learning one template for each class, averaging all the variations  
2. find the hyperplanes telling classes apart
**Hard cases for a linear classifier:**  
When can't draw a single linear line to separate classes
