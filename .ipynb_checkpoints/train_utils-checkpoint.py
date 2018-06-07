

"""
Concenvience module for pytorch training and visualization.
"""

import torch
import torch.nn as nn
from torch import optim
from pylab import *
from torch.utils.data import *
from IPython.display import *
from nb_figure import *
import os, time

def getTimeName():
    """Return the current time in format <day>-<month>_<hour><minute> for use in filenames."""
    from datetime import datetime
    t = datetime.now()
    return "{:02d}-{:02d}_{:02d}{:02d}".format(t.day,t.month,t.hour,t.minute)

def decorate_plot(axes):
    ax = axes[0][0]
    ax.set_title("Training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

def trainLoop(model, train_loader, optimizer, trainOnBatch, epochs=10, 
              save_every=1, save_dir=None, batch_interval=10, scheduler=None):
    
    run_dir = "run_{}".format(getTimeName())

    if save_dir==None:
        save_dir=type(model).__name__
    
    os.makedirs(os.path.join(save_dir,run_dir))
    
    batch_info = display(Markdown(''), display_id='batch_info')
    
    epoch_progress = ProgressBar(len(train_loader))
    progress_bar = display(epoch_progress,display_id='progress_bar')
    
    time_info = display(Markdown(''), display_id='time_info')
    
    checkpoint_info = display(Markdown(''), display_id='checkpoint_info')
    
    
    image_path = os.path.join(save_dir,run_dir,"loss_plot.png")
    loss_plot = NBFigure(image_path,decorate_fn=decorate_plot)
    loss_plot.set_xlim((1,None))
    loss_plot.set_ylim((0,None))
    loss_plot.display()
    loss_line = loss_plot.plotLine([0],[0])

    losses = []
    loss=0
    try:
        for e in range(epochs):
            i=0
            start = time.time()
            for batch in train_loader:
                loss = trainOnBatch(model,batch, optimizer)
                loss = loss.detach().cpu().numpy()
                i+=1
                if i%batch_interval==0:
                    toprint = "Epoch {}, batch {}, lr={:.6f}, loss={:.5f}".format(e+1,i,
                                optimizer.state_dict()['param_groups'][0]['lr'],loss)

                    batch_info.update(Markdown(toprint))
                    epoch_progress.progress = i
                    progress_bar.update(epoch_progress)

            losses.append(loss)
            loss_line.set_data(arange(e+1)+1,losses)
            loss_plot.update()

            t = time.time()-start
            toprint = "Last epoch took {:.2f} seconds".format(t)
            time_info.update(Markdown(toprint))
            
            if scheduler != None:
                scheduler.step(loss)

            if e%(save_every)==0:
                torch.save(model.state_dict(), 
                    os.path.join(
                        save_dir,
                        run_dir,                
                        "{}_epoch{:03d}_loss_{:.5f}.statedict".format(getTimeName(),e+1,loss)
                    )
                )

                toprint = """Saved model after epoch {} with 
                    loss={:.5f} on \n {}""".format(e+1,loss,time.ctime())
                checkpoint_info.update(Markdown(toprint))
                
    except KeyboardInterrupt:
        print("KeyboardInterrupt occured, saving raw model and losses")
    finally:
        torch.save(model,
              os.path.join(
                    save_dir,
                    run_dir,                
                    "{}_epoch{:03d}_loss_{:.5f}.model".format(type(model).__name__,e+1,loss)
                )
              )
        
        torch.save(losses,
              os.path.join(
                    save_dir,
                    run_dir,                
                    "{}_Losses_epoch{:03d}.list".format(type(model).__name__,e+1)
                )
              )
        
        return losses
    
    return losses