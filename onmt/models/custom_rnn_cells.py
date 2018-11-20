"""
Implementations of custom RNN Cells necessary for knowledge injection.
"""

import torch
import torch.nn as nn
import math


class HijackableLSTMCell(nn.Module):
    """
    LSTM Cell that separates forward pass into 2 steps:
        1 - computing new cell state value from inputs and previous hiddens
        2 - computing new hidden state output from inputs, previous hiddens and the updated cell state
    """
    def __init__(self, input_size, state_size):
        super(HijackableLSTMCell, self).__init__()
        self.input_size = input_size
        self.state_size = state_size
        # Weights have shape [number of gates * state size X input size + state size]
        # since gate output is computed from input
        self.weight_ih = nn.Parameter(torch.empty(4 * state_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * state_size, state_size))
        self.bias_ih = nn.Parameter(torch.empty(4 * state_size))
        self.bias_hh = nn.Parameter(torch.empty(4 * state_size))
        self._init_weights()

    def _init_weights(self):
        stdv = 1. / math.sqrt(self.input_size + self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    """
    Computes c_t+1 from x_t, h_t and c_t
        Return:
            c_t+1
            output gate (for computing h_t+1)
    """
    def compute_c(self, x, states, order=['i', 'f', 'g', 'o']):
        h, c = states

        # concatenate input and previous hidden state
        x = torch.cat([h, x], dim=1)  # [state size + input size X batch size]
        w = torch.cat([self.weight_hh, self.weight_ih], dim=1)
        b = self.bias_hh + self.bias_ih

        # compute forget, input and output gates and cell state addition
        gates = (torch.mm(x, w.transpose(1, 0)) + b).chunk(4, dim=1)
        input_gate = torch.sigmoid(gates[order.index('i')])
        forget_gate = torch.sigmoid(gates[order.index('f')])
        state_addition = torch.tanh(gates[order.index('g')])
        output_gate = torch.sigmoid(gates[order.index('o')])

        c_new = c * forget_gate + state_addition * input_gate

        return c_new, output_gate

    """
    Computes h_t+1 from c_t+1 and the output gate
        Return:
            h_t+1
    """
    def compute_h(self, c, output_gate):
        return torch.tanh(c) * output_gate

    """
    Compute cell's entire forward pass
    """
    def forward(self, input, states, order=['i', 'f', 'g', 'o']):
        # get new cell state and output gate
        c, output_gate = self.compute_c(input, states, order)

        # get new hidden state
        h = self.compute_h(c, output_gate)

        return h, c
