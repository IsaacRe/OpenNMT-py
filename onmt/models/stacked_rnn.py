"""
Implementation of ONMT RNN for Input Feeding Decoding

Modified for Version 1 functionality
Changes tagged with 'V1 Modification'
"""
import torch
import torch.nn as nn
from custom_rnn_cells import HijackableLSTMCell  # V1 Modification use HijcakableLSTMCell implementation as backend

#TODO
class HijackableLSTM(nn.Module):
    """
    Stacked LSTM same as below, but modified for compatibility with test-time knowledge injection
    """

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(HijackableLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(HijackableLSTMCell(input_size, rnn_size))
            input_size = rnn_size

    # V1 Modification: add method to compute returns from updated param, to be passed to knowledge sink

    def fast_forward(self, c_l_i, output_gate, start_layer, hidden_0, hidden_1):
        h_0, c_0 = hidden_0
        h_1, c_1 = hidden_1
        h_l_i = self.layers[start_layer].compute_h(c_l_i, output_gate)
        input_feed = h_l_i
        if start_layer + 1 != self.num_layers:
            input_feed = self.dropout(input_feed)
        for i, layer in enumerate(self.layers, start_layer+1):
            h_1_i, c_1_i = layer(input_feed, (h_0[i], c_0[i]))
            input_feed = h_1_i
            if i + 1 != self.num_layers:
                input_feed = self.dropout(input_feed)
            h_1 += [h_1_i]
            c_1 += [c_1_i]
        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input_feed, (h_1, c_1)

    # End Modification

    def forward(self, input_feed, hidden, knowledge_sink=None):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):

            # V1 Modification: on the final layer, get cell state and make it a parameter before
            # passing it to compute hidden output

            # currently only update on final RNN layer
            if i + 1 == self.num_layers and knowledge_sink:
                c_1_i, output_gate = layer.compute_c(input_feed, (h_0[i], c_0[i]))

                # save vars for fast_forward
                knowledge_sink.save_vars(output_gate=output_gate,
                                         start_layer=i,
                                         hidden_0=hidden,
                                         hidden_1=(h_1, c_1))
                # register cell state with knowledge sink as updatable parameter and pass kwargs to use
                # when knowledge sink calls fast forward
                c_1_i = knowledge_sink.register(c_1_i)

                h_1_i = layer.compute_h(c_1_i, output_gate)
            else:
                h_1_i, c_1_i = layer(input_feed, (h_0[i], c_0[i]))

            # End Modification

            input_feed = h_1_i
            if i + 1 != self.num_layers:
                input_feed = self.dropout(input_feed)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input_feed, (h_1, c_1)


class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input_feed, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input_feed, (h_0[i], c_0[i]))
            input_feed = h_1_i
            if i + 1 != self.num_layers:
                input_feed = self.dropout(input_feed)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input_feed, (h_1, c_1)


class StackedGRU(nn.Module):
    """
    Our own implementation of stacked GRU.
    Needed for the decoder, because we do input feeding.
    """

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input_feed, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input_feed, hidden[0][i])
            input_feed = h_1_i
            if i + 1 != self.num_layers:
                input_feed = self.dropout(input_feed)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)
        return input_feed, (h_1,)
