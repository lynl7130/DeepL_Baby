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
* real-time, speech-drivenm semantic highlighting

### Dataset
* extended version of the Places audio caption dataset(doubling)
* +10,000 captions for the ADE20k dataset

### Future Avenues
* expansion of the models to handle videos, environmental sounds, additional language.
* generate images given a spoken description or vice versa
* more focused datasets -> richer linguistic representations
* add dialog feedback loop to the models

