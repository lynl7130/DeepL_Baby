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
![](https://latex.codecogs.com/gif.latex?I_j): the output of the image branch of the network for the jth image.  
![](https://latex.codecogs.com/gif.latex?A_j): the output of the audio branch for the jth caption.  
![](https://latex.codecogs.com/gif.latex?S%28I%2C%20A%29): the similarity score between an image I and audio caption A.  
![](https://latex.codecogs.com/gif.latex?I_j%5E%7Bimp%7D): the jth randomly chosen imposter image.  
![](https://latex.codecogs.com/gif.latex?A_j%5E%7Bimp%7D): the jth randomly chosen imposter audio.  
![](https://latex.codecogs.com/gif.latex?%5Ceta): a margin hyperparameter.  
### Loss function:
![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bmatrix%7D%20L%20%3D%20%5Csum_%7Bj%3D1%7D%5EB%28max%28S%28I_j%2C%20A_j%5E%7Bimp%7D%29%20-%20S%28I_j%2C%20A_j%29&plus;%5Ceta%29%20%5C%5C%20&plus;max%280%2C%20S%28I_j%5E%7Bimp%7D%2C%20A_j%29%20-%20S%28I_j%2C%20A_j%29&plus;%5Ceta%29%29%20%5Cend%7Bmatrix%7D)  
