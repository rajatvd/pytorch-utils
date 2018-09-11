# PytorchUtils
Some utilities for building models in pytorch.

## Installation
First, clone this repository using   
`git clone https://github.com/rajatvd/PytorchUtils/`  

Then, `cd` into the directory named `package` and run the following command:  
`pip install .`  

## NBFigure
An extension of matplotlib figures to jupyter notebooks which are rendered using IPython Image displays. The plots are saved on disk and reloaded as images, allowing them to be easily updated dynamically. Useful for live loss plots.

## RNN modules
Currently contains:

* __WrappedLSTM__ :  a pytorch nn Module which wraps an input and output module around an lstm. The whole module now works solely with packed sequences, and padding is not required. (DEPRACTED in favor of WrappedRNN)
* __WrappedRNN__ : a module which wraps an input and output module around a general RNNBase instance.

## Train utils
Contains a Trainer class. It can be used to call a train loop with a model, DataLoader, optimizer, a trainOnBatch function and an epoch callback function to train a model for a given number of epochs. Automatically saves the model, displays a live animated metric plot, and a progress bar for each epoch.

![Example of the train loop util](train_util_example.PNG)

## Integration with sacred
Also has a `sacred_trainer` module which can be used with a sacred experiment to log metrics and artifacts. Provides a `loop` function which takes a `Run` instance and uses it to log batch and callback metrics, and also saves model weights as checkpoints. Combine it with the [visdom observer](https://github.com/rajatvd/VisdomObserver) to easily visualize your deep learning experiments as well.

## Todo

- [ ] Save best models to allow for easy reloading
- [ ] Make it easy to continue a run from a previous model checkpoint