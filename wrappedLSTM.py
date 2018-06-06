

"""
Module which wraps an input and output module around an LSTM.
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as utils

class WrappedLSTM(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, input_module=None, output_module=None):
        """
        lstm_input_size should equal input_module output size
        lstm_hidden_size should equal output_module input size
        """
        super(WrappedLSTM, self).__init__()
        
        self.input_module = input_module
        self.output_module = output_module
        
        self.lstm = nn.LSTM(input_size=lstm_input_size,hidden_size=lstm_hidden_size)
        
    def forward(self,hidden, *packed_input):
        """
        Applies input module to data in packed_inputs, 
        then applies the LSTM layers,
        Applied output module to output data of rnn,
        
        Returns packed output sequence and final hidden state of LSTM.
        """
        batch_sizes = packed_input[0].batch_sizes
        
        if self.input_module != None:
            rnn_input = self.input_module(*[p.data for p in packed_input])
            rnn_input = utils.PackedSequence(rnn_input,batch_sizes)
        else:
            rnn_input = packed_input[0]
            
        rnn_output, hidden = self.lstm(rnn_input,hidden)
        
        if self.output_module != None:
            output = self.output_module(rnn_output.data)
            output = utils.PackedSequence(output,batch_sizes)
        else:
            output = rnn_output
        
        return output, hidden