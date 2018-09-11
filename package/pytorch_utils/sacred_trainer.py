

"""
Convenience module for pytorch training and visualization. Uses sacred to
log experiments and visdom for visualization.
"""

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import *
import os, time
import traceback
from pytorch_utils.updaters import *
from tqdm import tqdm

def getTimeName():
    """Return the current time in format <day>-<month>_<hour><minute> for use in filenames."""
    from datetime import datetime
    t = datetime.now()
    return "{:02d}-{:02d}_{:02d}{:02d}".format(t.day,t.month,t.hour,t.minute)

def accuracy(scores, labels):
    """Return accuracy percentage. Assumes scores are in dim -1."""
    with torch.no_grad():
        total = scores.size()[0]
        pred = torch.argmax(scores, dim=-1)
        correct = (pred == labels).sum().cpu().numpy().astype('float32')
        acc = correct/total * 100
        return acc

def save_model(model, epoch, directory, metrics, filename=None):
    """Save the state dict of the model in the directory,
    with the save name metrics at the given epoch.

    epoch: epoch number(<= 3 digits)
    directory: where to save statedict
    metrics: dictionary of metrics to append to save filename
    filename: if a name is given, it overrides the above created filename

    Returns the save file name
    """
    # save state dict
    postfix = ""
    if filename is None:
        filename = f"epoch{epoch:03d}_{getTimeName()}_"
        postfix = "_".join([f"{name}{val:0.4f}" for name, val in metrics.items()])

    filename = os.path.join(directory, filename + postfix + ".statedict.pkl")

    torch.save(model.state_dict(), filename)
    print(f"Saved model at {filename}")
    
    return filename

def loop(_run,
     model,
     batch_metric_names,
     train_loader,
     trainOnBatch,
     optimizer,

     updaters,
     save_dir,

     callback=None,
     callback_metric_names=[],
     val_loader=None,

     epochs=10,
     save_every=1,
     start_epoch=1,

     **kwargs,
    ):
    """
    Arguments:
    -------------------
     _run: Sacred run instance

    model: Model instance to be trained

    trainOnBatch: Train on batch function with the following signature:
        trainOnBatch(model, batch, optimizer) -> tuple of batch metrics
    batch_metric_names: Names of the batch metrics returned by trainOnBatch

    train_loader: DataLoader which yields batches of training data
    optimizer: Optimizer instance which is passed to trainOnBatch
    updaters: List of running metric updaters which aggregate batch metrics per epoch.
        They should be generator functions which return the running value when
        .send is called with a batch value.
    save_dir: Top level directory in which to save configs, checkpoints and metrics
    of each run

    callback=None : Optional callback function, should have the following signature:
        callback(model, val_loader) -> tuple of callback metrics
        Usually used for calculating validation metrics.
    callback_metric_names=[]: Names of the above returned callback metrics
    val_loader=None: DataLoader for validation data.
    """
    run_dir = os.path.join(save_dir, str(_run._id))
    os.makedirs(run_dir, exist_ok=True)

    try:
        for e in range(start_epoch,start_epoch+epochs,1):

            i=0

            upds = [updater() for updater in updaters]
            [next(u) for u in upds]

            t = tqdm(train_loader, desc=f'Epoch: {e}')
            for i,batch in enumerate(t):
                # Perform train step
                batch_metrics = trainOnBatch(model, batch, optimizer)

                # Update running metrics
                batch_metrics = [upds[i].send(b_metric)
                                    for i,b_metric in enumerate(batch_metrics)]


                postfix = " ".join([f"{name}={value:.4f}"
                    for name, value in zip(batch_metric_names, batch_metrics)])

                t.set_postfix_str(postfix)


            batch_metrics_dict = dict(zip(batch_metric_names, batch_metrics))

            # execute callback
            callback_metrics = ()
            if callback != None:
                callback_metrics = callback(model,
                                    val_loader=val_loader,
                                    batch_metrics_dict=batch_metrics_dict)

            callback_metrics_dict = dict(zip(callback_metric_names,
                                             callback_metrics))
            mets = {**callback_metrics_dict, **batch_metrics_dict}

            if len(callback_metrics)!=0:
                cb_info = "Callback metrics: " + " ".join([f"{name}={val:.6f}"
                                for name,val in callback_metrics_dict.items()])
                print(cb_info)

            # log metrics
            for name,val in zip(batch_metric_names, batch_metrics):
                _run.log_scalar(name, val, e)
            for name,val in zip(callback_metric_names, callback_metrics):
                _run.log_scalar(name, val, e)

            # Checkpointing
            fname = save_model(model, e, run_dir, mets,
                               filename='latest')
            if e%(save_every)==0:
                fname = save_model(model, e, run_dir, mets)
#                 _run.add_artifact(fname)


    except:
        print("Exception occured, saving optimizer and model")
        traceback.print_exc()
    finally:

        # save optimizer state
        fname = os.path.join(
                    run_dir,
                    f"optimizer_state_epoch{e:03d}.statedict.pkl"
                )
        torch.save(optimizer.state_dict(),fname)
#         _run.add_artifact(fname)

        # save model dict
        fname = save_model(model, e, run_dir, mets)
#         _run.add_artifact(fname)

CLASSIFICATION = dict(
    batch_metric_names=['loss', 'acc'],
    callback_metric_names=['val_loss', 'val_acc'],
    updaters=[averager, averager],
)