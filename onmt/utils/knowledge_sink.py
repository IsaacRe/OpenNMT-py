from torch.nn import Parameter, CrossEntropyLoss
import torch
from copy import deepcopy


class KnowledgeSink(object):
    """
    Abstract module class to handle manipulation of network activations during test-time
    """

    def __init__(self):
        super(KnowledgeSink, self).__init__()
        self._nodes = {}
        self._modules = []
        self.update_var = None

    """
    Method to remove references to scope variables and update variables so that garbage collecter
    may free up space
    """
    def _clear_references(self):
        self._nodes = {}
        self._modules = []
        self.update_var = None
        # clear all unreferenced memory space from the gpu
        torch.cuda.empty_cache()

    """
    Forward pass registers the activation and returns parameterized version
    """
    def register(self, x):
        param = Parameter(x.detach().data.clone())
        self.update_var = param
        return param

    """
    Return updated values for the activation making use of all registered nodes and the extra knowledge
    Inheriting classes must implement _correct(activation, knowledge)
    """
    def correct(self, output, knowledge):
        assert self.update_var is not None
        output = self._correct(output, knowledge)

        # when done with updates, remove all references to scope variables
        self._clear_references()

        return output

    """
    Save variables for use in fast_forward calls
    """
    def save_vars(self, module, **kwargs):
        if module not in self._nodes:
            self._modules += [module]
            self._nodes[module] = {}
        # try to perform deepcopy for all saved vars so that context is unaffected by remainder of forward pass
        save_kwargs = {}
        for k, v in kwargs.items():
            if type(v) == torch.Tensor:
                v = v.data
            try:
                v = deepcopy(v)
            except:
                pass
            save_kwargs[k] = v
        self._nodes[module].update(save_kwargs)


class StateUpdater(KnowledgeSink):
    """
    Module to handle update of hidden LSTM states based on gradient of cross-entropy loss between
        log probability output and target output

            step: the step size of the update, update will be: state -= gradient * step
            step: the step of the gradient update
            make_loss: function to compute the loss given a hidden output to be backpropped to cell states
                            and used in the update
    """
    def __init__(self, step):
        super(StateUpdater, self).__init__()
        self._step = step
        # we want to keep gradients separated w.r.t. each sample in the batch, so we do not reduce loss to a scalar
        self._make_loss = CrossEntropyLoss(reduction='none')

    """
    Compute loss from logits and target, backprop to get update, update state and compute forward
        pass again from state node and return cell state, hidden state and logits. Current outputs are
        averaged over all forward-propagated signals from each of the registered states
    """
    def _correct(self, output, targets):
        loss = self._make_loss(output, targets)

        # clear gradient for all states being updated
        if self.update_var.grad is not None:
            self.update_var.grad.zero_()

        # necessary to specify gradient of output w.r.t. output when loss is not scalar
        # TODO implement on gpu
        #start_grads = torch.ones(loss.shape)  # .cuda(torch.cuda.current_device())
        # retain_graph only necessary when we begin to apply multiple updates
        # TODO implement for batch training
        loss = torch.sum(loss)
        loss.backward()

        assert self.update_var.grad is not None

        # Perform update for registered cell_state and follow fast_forward methods to compute new output
        self.update_var.data -= self.update_var.grad * self._step
        new_output = [self.update_var]
        for module in self._modules[::-1]:
            kwargs = self._nodes[module]
            new_output = module.fast_forward(*new_output, **kwargs)

        return new_output

