# PytorchUtils
Some utilities for building models in pytorch:

## NBFigure
An extension of matplotlib figures to jupyter notebooks which are rendered using IPython Image displays. The plots are saved on disk and reloaded as images, allowing them to be easily updated dynamically. Useful for live loss plots.

## RNN modules
Currently contains:

* __WrappedLSTM__ :  a pytorch nn Module which wraps an input and output module around an lstm. The whole module now works solely with packed sequences, and padding is not required.

## Train utils
Contains a train loop util which can be called with a model, DataLoader, optimizer and a trainOnBatch function to train a model for a given number of epochs. Automatically saves the model, displays a live animated loss plot, and a progress bar for each epoch.

![Example of the train loop util](train_util_example.PNG)