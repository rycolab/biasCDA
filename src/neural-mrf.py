import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from mrf_op import MRF_NN, MRF_Lin
from Data import Data
import os
from tqdm import tqdm



class NeuralMRF(nn.Module):
    """ neural MRF """

    def __init__(self, data, out_dir, linear, use_v1, hack_v2):
        super(NeuralMRF, self).__init__()

        self.data = Data(data + "-train.conllu", data + "-dev.conllu", data + "-test.conllu", use_v1, hack_v2)
        self.num_pos = self.data.num_pos()
        self.num_labels = self.data.num_labels()
        self.num_tags = self.data.num_tags()

        self.out_dir = out_dir

        self.linear = linear

        if self.linear:
            self.mrf = MRF_Lin(self.data.tags)
            self.register_parameter('psi', None)
            self.psi = nn.Parameter(
                torch.randn(self.num_pos, self.num_pos, self.num_labels, self.num_tags, self.num_tags,
                            dtype=torch.float64))
        else:
            self.mrf = MRF_NN(self.data.tags)
            self.n = 3

            self.register_parameter('pos', None)
            self.register_parameter('labels', None)
            self.register_parameter('psi_2', None)
            self.register_parameter('W', None)

            self.pos = nn.Parameter(torch.randn(self.num_pos, self.n, dtype=torch.float64))
            self.labels = nn.Parameter(torch.randn((self.num_labels, self.n), dtype=torch.float64))

            self.psi_2 = nn.Parameter(torch.randn(
                (self.n * 3, self.num_tags, self.num_tags), dtype=torch.float64))
            self.W = nn.Parameter(torch.randn(
                (self.n * 3, self.n * 3), dtype=torch.float64))

    def forward(self, sentence):
        """ computation of the log-likelihood with pytorch """
        self.mrf.sentence = sentence
        if self.linear:
            return self.mrf(self.psi)
        else:
            return self.mrf(self.pos, self.labels, self.W, self.psi_2)

    def fit(self, epochs=100, precision=1e-5):
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=0.001)

        def step():
            """ step in the optimization """
            train_loss = dev_loss = 0
            print("  Optimizing parameters on training data loss")
            for sentence in tqdm(self.data, total=len(self.data.train)):
                self.optimizer.zero_grad()
                loss = self.forward(sentence)
                train_loss += loss
                loss.backward()
                self.optimizer.step()
                del loss
            print("  Calculating dev loss")
            for sentence in tqdm(self.data.dev, total=len(self.data.dev)):
                dev_loss += self.forward(sentence)
            return train_loss / len(self.data.train), dev_loss / len(self.data.dev)
        for i in range(epochs):
            print("Computing epoch", i + 1, "...")
            # Do optimization step
            train_loss, dev_loss = step()
            # Save current parameters
            file = os.path.join(self.out_dir, "psi_" +
                                str(round(train_loss[0].item(), 6)) + "_" +
                                str(round(dev_loss[0].item(), 6)) + "_epoch" + str(i + 1) + ".pt")
            if self.linear:
                torch.save(self.psi, file)
            else:
                pos2 = self.pos.repeat((1, self.num_labels)).view(-1, self.n).repeat(self.num_pos, 1)
                pos1 = self.pos.repeat((1, self.num_pos * self.num_labels)).view(-1, self.n)
                labels = self.labels.repeat((self.num_pos * self.num_pos, 1))
                psi_1 = torch.cat([pos1, pos2, labels], 1).reshape((self.num_pos, self.num_pos, self.num_labels, self.n * 3))

                tanh = nn.Tanh()
                psi = torch.tensordot(tanh(torch.tensordot(psi_1, self.W, 1)), self.psi_2, 1)
                torch.save(psi, file)
                del pos2, pos1, labels, psi_1, psi

            print("Completed epoch", i + 1)
            print("    Training loss:", train_loss[0].item())
            print("    Dev loss:     ", dev_loss[0].item())
            if i > 0 and prev_loss - train_loss < precision:
                break
            prev_loss = train_loss


if __name__ == "__main__":
    from argparse import ArgumentParser

    p = ArgumentParser()
    p.add_argument('--data', required=True, type=str)
    p.add_argument('--out_dir', required=True, type=str)
    p.add_argument('--use_v1', default=False, action='store_true')
    p.add_argument('--hack_v2', default=False, action='store_true')
    p.add_argument('--linear', default=False, action='store_true')

    args = p.parse_args()

    nmrf = NeuralMRF(args.data, args.out_dir, args.linear, args.use_v1, args.hack_v2)
    nmrf.fit()
