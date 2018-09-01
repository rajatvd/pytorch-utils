
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import *
from pytorch_utils.updaters import *
import pytorch_utils.sacred_trainer as st
import torch.nn.functional as F
from sacred import Experiment
from sacred.observers import FileStorageObserver
from visdom_observer.visdom_observer import VisdomObserver
from torchvision.datasets import *
from torchvision.transforms import *

IPY=True
try:
    get_ipython()
except:
    IPY=False

ex = Experiment('mnistclassifier_example', interactive=IPY)
save_dir = 'MnistClassifier'
ex.observers.append(VisdomObserver())
ex.observers.append(FileStorageObserver.create(save_dir))

@ex.config
def dataset_config():
    val_split=0.1
    batch_size=32

@ex.capture
def make_dataloaders(val_split, batch_size):

    # get dataset
    total_train_mnist = MNIST('Z:/MNIST',download=True,transform=ToTensor())
    test_mnist = MNIST('Z:/MNIST',train=False,transform=ToTensor())

    training_number = len(total_train_mnist)

    train_num = int(training_number*(1-val_split))
    val_num = int(training_number*val_split)

    train_mnist, val_mnist = torch.utils.data.dataset.random_split(total_train_mnist,
                    [train_num, val_num])


    # split into data loaders
    train_loader = DataLoader(train_mnist, batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_mnist, batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_mnist, batch_size=batch_size,shuffle=True)

    return train_loader, val_loader, test_loader

@ex.config
def model_config():
    hidden_size = 400 # hidden units in hidden layer
    output_size = 10 # number of output labels

class MnistClassifier(nn.Module):
    """A simple two layer fully connected neural network"""

    def __init__(self, hidden_size, output_size):
        super(MnistClassifier, self).__init__()

        self.hidden_layer = nn.Linear(28*28, hidden_size)

        self.output_layer = nn.Linear(hidden_size, output_size)


    def forward(self, input):
        input = input.view(-1,28*28)

        out = F.relu(self.hidden_layer(input))
        out = self.output_layer(out)

        return out

@ex.capture
def make_model(hidden_size, output_size):
    return MnistClassifier(hidden_size, output_size)

@ex.config
def optimizer_config():
    lr=0.001 # learning rate
    weight_decay=0 # l2 regularization factor

@ex.capture
def make_optimizer(model, lr, weight_decay):
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def classification_train_on_batch(model,batch,optimizer):

    # batch is tuple containing (tensor of images, tensor of labels)
    outputs = model(batch[0]) # forward pass

    # compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs,batch[1])

    # backward pass and weight update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # compute and return metrics
    loss = loss.detach().cpu().numpy()
    acc = st.accuracy(outputs, batch[1])

    return loss, acc

def classification_callback(model, val_loader, batch_metrics_dict):
    with torch.no_grad(): # dont compute gradients
        criterion = nn.CrossEntropyLoss()

        model.eval() # eval mode

        batches = len(val_loader)
        loss=0
        acc=0
        for batch in val_loader:
            outputs = model(batch[0])
            loss += criterion(outputs,batch[1])
            acc += st.accuracy(outputs,batch[1])

        # find average loss and accuracy over whole vaildation set
        loss/= batches
        acc /= batches

        model.train() # go back to train mode

        # return metrics
        return loss.cpu().numpy(), acc

@ex.config
def train_config():
    epochs=10
    save_every=1 # epoch interval with which to save models

@ex.command
def train(_run):

    train_loader, val_loader, test_loader = make_dataloaders()
    model = make_model()
    optimizer = make_optimizer(model)

    st.loop(
        _run=_run,
        model=model,
        optimizer=optimizer,
        save_dir=save_dir,
        trainOnBatch=classification_train_on_batch,
        train_loader=train_loader,
        val_loader=val_loader,
        callback=classification_callback,
        **_run.config,
        **st.CLASSIFICATION,
    )

if __name__ == '__main__':
    ex.run_commandline()