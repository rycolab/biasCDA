import torch
from torch.autograd import Function
import torch.nn as nn
from model import Model


class MRF(Function):
    """
    Class to compute dpsi of a given sentence
    """
    def __init__(self, model, sentence):
        super(MRF, self).__init__()
        self.sentence = sentence
        self.model = model

    def forward(self, psi):
        """
        :param psi: psi potentials to use in computing -log Pr(T|m)
        :return: -log Pr(T|m)
        """
        self.save_for_backward(psi)
        val = -self.model.log_prob(self.sentence.T, self.sentence.pos, self.sentence.m, psi)
        return torch.Tensor([val])

    def backward(self, grad_output):
        """
        :param grad_output: N/A
        :return: gradient of -log Pr(T|m) wrt psi
        """
        psi = self.saved_tensors[0]
        dpsi = -self.model.dlog_prob(self.sentence.T, self.sentence.pos, self.sentence.m, psi)
        del psi
        return dpsi


class MRF_NN(torch.nn.Module):
    """
    Class to initialize belief propagation model with neural parametrization of psi
    """
    def __init__(self, tags, sentence=None):
        super(MRF_NN, self).__init__()
        self.model = Model(tags)
        self.sentence = sentence
        
    def forward(self, pos, labs, W, psi_2):
        """
        :param pos: pos tags parameters
        :param labs: dependency label parameters
        :param W: weight matrix
        :param psi_2: message parameters
        :return: application of model to current sentence
        """
        num_labels, num_pos, n = len(labs), len(pos), len(pos[0])
        pos2 = pos.repeat((1, num_labels)).view(-1, n).repeat(num_pos, 1)
        pos1 = pos.repeat((1, num_pos * num_labels)).view(-1, n)
        labels = labs.repeat((num_pos * num_pos, 1))

        psi_1 = torch.cat([pos1, pos2, labels], 1).reshape((num_pos, num_pos, num_labels, n * 3))
        tanh = nn.Tanh()
        psi = torch.tensordot(tanh(torch.tensordot(psi_1, W, 1)), psi_2, 1)
        del pos2, pos1, labels, psi_1
        return MRF(self.model, self.sentence)(psi)


class MRF_Lin(torch.nn.Module):
    """
    Class to initialize belief propagation model with linear parametrization of psi
    """
    def __init__(self, tags, sentence=None):
        super(MRF_Lin, self).__init__()
        self.model = Model(tags)
        self.sentence = sentence

    def forward(self, psi):
        """
        :param psi: psi parameters
        :return: application of model to current sentence
        """
        return MRF(self.model, self.sentence)(psi)
