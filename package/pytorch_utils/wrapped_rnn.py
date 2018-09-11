"""
Module which wraps an input and output module around an RNNBase module. (Can be
an RNN, GRU, or vanilla RNN)
"""

import torch.nn as nn
import torch.nn.utils.rnn as utils

class WrappedRNN(nn.Module):
    """Wraps an input and output module around an RNN"""
    def __init__(self,
                 mode,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0,
                 bidirectional=False,
                 input_module=None,
                 output_module=None):
        """
        Args:
            mode: The type of RNN to use. One of ['LSTM', 'GRU',
                'RNN_TANH', 'RNN_RELU']
            input_size: The number of expected features in the input `x`
            hidden_size: The number of features in the hidden state `h`
            num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
                would mean stacking two RNNs together to form a `stacked RNN`,
                with the second RNN taking in outputs of the first RNN and
                computing the final results. Default: 1
            bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
                Default: ``True``
            batch_first: If ``True``, then the input and output tensors are provided
                as (batch, seq, feature)
            dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
                RNN layer except the last layer, with dropout probability equal to
                :attr:`dropout`. Default: 0
            bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``

            input_module: A module to apply on the packed input data before feeding into
                the RNN

            output_module A module to apply to the packed output data of the RNN

        Inputs: hidden_state, packed_inputs
            - **packed_inputs** A list of input `PackedSequence`s whose data are fed into the
              input module (the data list is exapnded using * and fed).
              See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
              :func:`torch.nn.utils.rnn.pack_sequence` for details.
            - **hidden_state** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
              containing the initial hidden state for each element in the batch.
            - An additional cell state is also needed if mode is 'LSTM'


        Outputs: output, hidden_state
            - **output** A packed variable length sequence with the data from the output_module.
              See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
              :func:`torch.nn.utils.rnn.pack_sequence` for details.
            - **hidden_state** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
              containing the hidden state for `t = seq_len`
            - An additional cell state is also returned if mode is 'LSTM'
        """
        super().__init__()

        self.input_module = input_module
        self.output_module = output_module

        self.rnn = nn.RNNBase(mode,
                              input_size,
                              hidden_size,
                              num_layers,
                              bias,
                              batch_first,
                              dropout,
                              bidirectional)

    def forward(self, hidden, *packed_inputs):
        """
        Applies input module to data in packed_inputs,
        then applies the RNN layers,
        Applies output module to output data of rnn,

        Returns packed output sequence and final hidden state of RNN.
        """
        batch_sizes = packed_inputs[0].batch_sizes

        if self.input_module != None:
            rnn_input = self.input_module(*[p.data for p in packed_inputs])
            rnn_input = utils.PackedSequence(rnn_input, batch_sizes)
        else:
            rnn_input = packed_inputs[0]

        rnn_output, hidden = self.rnn(rnn_input, hidden)

        if self.output_module != None:
            output = self.output_module(rnn_output.data)
            output = utils.PackedSequence(output, batch_sizes)
        else:
            output = rnn_output

        return output, hidden
