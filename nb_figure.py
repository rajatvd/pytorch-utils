
"""Module which extends matplotlib figures to notebooks. 
Figures are displayed as loaded images using IPython display."""

import matplotlib.pyplot as plt
from IPython.display import *
import numpy as np

class NBFigure():
    """An extension of matplotlib figure to work with jupyter notebook's display module.
    Works by storing the figure as an image on disk and loading it to display on updating.
    """
    def __init__(self,image_path,nrows=1,ncols=1,**subplot_kwargs):
        """
        image_path: the filename of the stored figure image
        
        Rest of the arguments are indentical to subplots' arguments
        """
        
        self.nrows=nrows
        self.ncols=ncols
        self.fig, self.axes = plt.subplots(nrows,ncols,**subplot_kwargs)
        
        self.axes=self.axes.reshape(nrows,ncols)
        plt.close(self.fig)
        
        self.image_path = image_path
        self.fig.savefig(image_path, bbox_inches='tight')
        self.disp = None
        
        self.xlims = [[(None,None) for j in range(ncols)] for i in range(nrows)]
        self.ylims = [[(None,None) for j in range(ncols)] for i in range(nrows)]
        
    def update_lims(self):
        """
        Updates xlims
        """
        for i in range(self.nrows):
            for j in range(self.ncols):
                ax = self.axes[i][j]
                ax.relim()
                ax.autoscale()
                ax.set_xlim(self.xlims[i][j])
                ax.set_ylim(self.ylims[i][j])
                
    def set_xlim(self,xlim,row=0,col=0):
        self.xlims[row][col]=xlim

    def set_ylim(self,ylim,row=0,col=0):
        self.ylims[row][col]=ylim
        
    def display(self):
        """Create a new display of the figure"""
        self.disp = display(Image(self.image_path),display_id=str(id(self)))
    
    def update(self):
        """Update the lims and update all display instances"""
        self.update_lims()
        self.fig.savefig(self.image_path, bbox_inches='tight')
        self.disp.update(Image(self.image_path))
        
    def plotLine(self,x,y,row=0,col=0,**plot_kwargs):
        """Plot a line and return the Line2D object"""
        l,= self.axes[row][col].plot(x,y,**plot_kwargs)
        return l