# Jointly Discovering Visual Objects and Spoken Words from Raw Sensory Input
[paper](https://arxiv.org/pdf/1804.01452.pdf)

## Summary
* **Associate**: segments of spoken audio captions + semantically relevant portions of images
* Audio-visual associative localizations are by-product in image-audio retrievel task.
* Model operates on image pixels and speech waveform.
* Places 205 and ADE20k datasets.
  
### audio-visual "matchmap" NN
learning the semantic correspondences between speech frames and image pixels.
* semantic image/spoken caption search
* speech-prompted object localization
* audio-visual clustering
* concept discovery
* real-time, speech-driven semantic highlighting

### Dataset
* extended version of the Places audio caption dataset(doubling)
* +10,000 captions for the ADE20k dataset

### Future Avenues
* expansion of the models to handle videos, environmental sounds, additional language.
* generate images given a spoken description or vice versa
* more focused datasets -> richer linguistic representations
* add dialog feedback loop to the models

## Introduction
### Task?
Jointly learn spoken language and visual perception given raw speech audio and images.

### Solution
Models capable of jointly discovering words in raw speech audio, objects in raw images, and associating them with one another.

### What has been done: written text <-> vision!  
written text: has been segemented and clustered.  
**Two problems to solve in this paper:**  
1. segmenting and clustering the raw speech signal into discrete words
2. visual object discovery in images

### What has been done: sounds <-> vision!
**Problems in this work:**  
* portions of the speech signal that refers to objects are shorter.
* number of categories is much larger.  

### premise of this paper
* NN: image+audio->highlight the relevant regions of the image as described in speech
* no speech recognition/transcription, no conventional object detection/recognition
* perform semantic retrieval at the whole-image and whole-caption level
* detection and localization of both visual objects and spoken words emerges as a by-product of this training

## Prior Work
### Visual Object Recognition and Discovery
* bounding box annotation in training data
* weakly-supervised or unsupervised visual object localization
* unsupervised visual object discovery
  
### Unsupervised Speech Processing
ASR systems for now: expensive for supervision! -> reduce supervision is needed.  
-> segmentation and clustering algorithms:  
1. divide a collection of spoken utterances at the boundaries of phones or words
2. group together segments which capture the same underlying unit.  
  
-> popular approaches:
* dynamic time warping
* Bayesian generative models of the speech signal

### Fusion of Vision and Language
popular tasks:
* image captioning
* visual question answering
* multimodal dialog
* text-to-image generation
  
Most of them: representing natural language with text.  
Rising: learn directly from the speech signal:
* correspondences between images of objects and the outputs of a supervised phoneme recognizer
* showed that semantic correspondences could be learned between images and speech waveforms
* proved that linguistic units approximating phonemes and words can be implicitly learned by models
* **This paper introduces "matchmap" NNs: get semantic alignments between acoustic frames and image pixels**

### Fusion of Vision and Sounds
Some focused on integrating other acoustic signals.  
**This paper concentrates on speech and word discovery.**  
Combining speech and ambient sounds -> opportunities!


## Spoken Captions Dataset -- Places Audio Caption Dataset
~200,000 recordings describing images from Places 205.  
402,385 image/caption pairs for training + 1,000 pairs for validation.  
ADE20k dataset + ASR: for localize objects and words.  
Google ASR engine: in place of ground truth.  

## Models
[Two-branch "matchmap" Network](https://arxiv.org/pdf/1804.01452.pdf#page=6)  
Different from prior work by learning representations that are distributed both spatially and temporally, enabling our models to directly co-localize within both modalities.  
  
![](https://latex.codecogs.com/gif.latex?I_j): the output of the image branch of the network for the jth image.  
![](https://latex.codecogs.com/gif.latex?A_j): the output of the audio branch for the jth caption.  
![](https://latex.codecogs.com/gif.latex?S%28I%2C%20A%29): the similarity score between an image I and audio caption A.  
![](https://latex.codecogs.com/gif.latex?I_j%5E%7Bimp%7D): the jth randomly chosen imposter image.  
![](https://latex.codecogs.com/gif.latex?A_j%5E%7Bimp%7D): the jth randomly chosen imposter audio.  
![](https://latex.codecogs.com/gif.latex?%5Ceta): a margin hyperparameter.  
  
### Loss function:
![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bmatrix%7D%20L%20%3D%20%5Csum_%7Bj%3D1%7D%5EB%28max%28S%28I_j%2C%20A_j%5E%7Bimp%7D%29%20-%20S%28I_j%2C%20A_j%29&plus;%5Ceta%29%20%5C%5C%20&plus;max%280%2C%20S%28I_j%5E%7Bimp%7D%2C%20A_j%29%20-%20S%28I_j%2C%20A_j%29&plus;%5Ceta%29%29%20%5Cend%7Bmatrix%7D)  
Our models are trained to optimize a ranking-based cirterion, such that images and captions that belong together are more similar in the embedding space than mismatched image/caption pairs.  
  
### Image Modeling
##### Preprocessing:
1. resizing the smallest dimension to 256 pixels
2. taking a random 224x224 crop(the center crop is taken for validation)
3. normalizing the pixels according to a global mean and variance

#### New in VGG16 network:  
* do not require pre-training
* fc layer -> lose associations between any neuron above conv5 and the spatially localized stimulus which was responsible for its output -> **do not include fc layers!**
* apply a 3x3, 1024 channel conv layer(no nonlinearity!) to the final 14x14, 512 channel feature map

### Audio Caption Modeling
output a feature map across the audio during training.  

### Joining the Image and Audio Branches
![](https://latex.codecogs.com/gif.latex?I): output feature map of the image network branch.  
![](https://latex.codecogs.com/gif.latex?A): output feature map of the audio network branch.  
![](https://latex.codecogs.com/gif.latex?I%5Ep): globally average-pooled I.  
![](https://latex.codecogs.com/gif.latex?A%5Ep): globally average-pooled A.  
**"matchmap" tensor** between an image and an audio caption:  
![](https://latex.codecogs.com/gif.latex?M_%7Br%2Cc%2Ct%7D%20%3D%20I_%7Br%2Cc%2C%3A%7D%5ETA_%7Bt%2C%3A%7D)  

#### SISA
The average similarity between all audio frames and all image regions:  
![](https://latex.codecogs.com/gif.latex?SISA%28M%29%20%3D%20%5Cfrac%7B1%7D%7BN_rN_cN_t%7D%5Csum_%7Br%3D1%7D%5E%7BN_r%7D%5Csum_%7Bc%3D1%7D%5E%7BN_c%7D%5Csum_%7Bt%3D1%7D%5E%7BN_t%7DM_%7Br%2Cc%2Ct%7D)  

#### MISA
matches each frame of the caption with the most similar image patch, then averages over the caption frames:  
![](https://latex.codecogs.com/gif.latex?MISA%28M%29%20%3D%20%5Cfrac%7B1%7D%7BN_t%7D%5Csum_%7Bt%3D1%7D%5E%7BN_t%7Dmax_%7Br%2Cc%7D%28M_%7Br%2Cc%2Ct%7D%29)

#### SIMA
matches each image region with only the audio frame with the highest similarity to that region:
![](https://latex.codecogs.com/gif.latex?SIMA%28M%29%20%3D%20%5Cfrac%7B1%7D%7BN_rN_c%7D%5Csum_%7Br%3D1%7D%5E%7BN_r%7D%5Csum_%7Bc%3D1%7D%5E%7BN_c%7Dmax_t%28M_%7Br%2Cc%2Ct%7D%29)
