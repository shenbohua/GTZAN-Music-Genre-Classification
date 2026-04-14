# COMP6252 Coursework 1 - Music Genre Classification

## Overview
The goal of this assignment is to build different classifiers for the music genre dataset GTZAN using different network architectures, report their performance, and provide a brief discussion regarding the results.

## Dataset
The GTZAN dataset is used for this project and can be downloaded from Kaggle: [GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzandataset-music-genre-classification)

The dataset includes:
* Audio samples for 10 different music genres (e.g., jazz, classical, etc.).
* Visual representations constructed using MEL spectrograms for each audio sample.

## Neural Architectures
The following six architectures are implemented sequentially as `Net1` to `Net6`:

1. **Net1:** A fully connected network with two hidden layers.
2. **Net2:** A convolutional network (based on the provided Figure 1) with custom choice of parameters.
3. **Net3:** A convolutional network modified from Figure 1 by adding a batch normalisation layer.
4. **Net4:** The same architecture as Net3, but trained with the RMSProp optimiser.
5. **Net5:** An RNN network with LSTMs.
6. **Net6:** The same architecture as Net5, together with GANs generating audio samples to augment the training data (generating the same number of audio samples as the original training set).

## Implementation Details

### For Architectures 1 - 4 (MEL Spectrogram Images)
* **Image Preprocessing:** Resize the images to 180x180 using `torchvision.transforms.Resize` when loading.
* **Data Splitting:** Use PyTorch to randomly split the dataset into Training (70%), Validation (20%), and Test (10%) datasets.
* **Training Epochs:** Run the training for 50 epochs and 100 epochs.

### For Architectures 5 - 6 (Audio Samples)
* **Input Data:** Just use the audio samples directly.
* **Data Splitting:** Use PyTorch to randomly split the original dataset into Training (70%), Validation (20%), and Test (10%) datasets.
* **Training Epochs:** Run the training for a certain number of epochs (e.g., until convergence according to your stopping criterion).


## Environment Dependencies
```txt
Python==3.12.11
torch==2.10.0
torchaudio==2.10.0
torchvision==0.25.0
numpy==2.3.3
pandas==2.3.3
seaborn==0.13.2
matplotlib==3.10.6
scikit-learn==1.7.2
scipy==1.16.2
pillow==11.3.0
```


## Quick Start
**Environment Setup**
```txt
# Clone the project
git clone https://github.com/shenbohua/GTZAN-Music-Genre-Classification.git
cd GTZAN-Music-Genre-Classification

# Install dependencies
pip install -r requirements.txt

# Run code
python main.py
```


## License
This project is only used for academic research and learning, and follows the MIT open source license. For details, please refer to the LICENSE file.