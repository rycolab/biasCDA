from belief_propagation import belief_propagation, calculate_gradient, calculate_belief_sum,\
    max_product, get_best_tags
from itertools import product
from utils.math import logsumexp
import torch as tr


class Model(object):
    """
    Class to get distribution p(m|T)
    """
    def __init__(self, tags):
        """
        Model is initialized with tag model m

        :param tags: list of possible tags
        for tag i are given by tag_model[i]
        """
        self.tags = tags

    """
    Tagset related functions
    """
    def tag_size(self):
        """
        :return: number of possible tag sequences
        """
        return len(self.tags)

    def get_tags(self):
        """
        :return: possible tags
        """
        return self.tags

    def get_all_tag_seq(self, n):
        """
        Get all the tag combinations possible for a tree of length n

        :param n: number of nodes in the tree
        :return: list of all possible length-n sequences of tags
        """
        tags = list(product(self.tags, repeat=n))
        return tags

    def get_tag_index(self, m):
        """
        Get index of a tag sequence m in self.tags

        :param m: tag sequence
        :return: i such that self.tags[i] == m
        """
        return self.tags.index(m)

    def get_tag(self, idx):
        return self.tags[idx]

    def get_psi_score(self, psi, pos1, pos2, lab, m1, m2):
        """
        Given two tags and a label, return the psi factor of the two tag sequences

        :param psi: psi potentials
        :param pos1: pos tag 1 of factor
        :param pos2: pos tag 2 of factor
        :param lab: label of factor
        :param m1: first tag sequence
        :param m2: second tag sequence
        :return: psi(m_i, m_j, l)
        """
        i, j = self.get_tag_index(m1), self.get_tag_index(m2)
        return psi[pos1, pos2, lab, i, j]

    def get_phi_score(self, phi, i, m):
        """
        Given a word index and a tag, return the corresponding phi factor

        :param phi: phi potential
        :param m: tag sequence
        :return: phi(m)
        """
        m_i = self.get_tag_index(m)
        return phi[i, m_i]

    def create_phi(self, T, pos, m, alpha=1):
        """
        Create phi factors for a given tree
        :param T: tree
        :param pos: pos sequence
        :param m: tag sequence
        :param for_train: True if phi will be used for training
        :return:
        """
        phi = tr.zeros((len(T), self.tag_size()), dtype=tr.float64)
        for i, _, _ in T:
            m_i = self.get_tag_index(m[i - 1])
            phi[i - 1, m_i] = alpha
        return phi

    def log_score(self, T, pos, m, psi, phi):
        """
        Calculate the (log) agreement of a list of tags for a given tree

        :param T: tree
        :param pos: list of pos tags
        :param m: list of tags
        :param psi: psi potentials
        :param phi: phi potentials
        :return: (log) agreement of tags m for tree T
        """
        log_score = 0
        for (i, j, lab) in T:
            m_i = m[i - 1]
            pos1 = pos[i - 1]
            log_score += self.get_phi_score(phi, i - 1, m_i)
            if j != 0:
                m_j = m[j - 1]
                pos2 = pos[j - 1]
                log_score += self.get_psi_score(psi, pos1, pos2, lab, m_i, m_j)
        return log_score

    def dlog_score(self, T, pos, m, psi):
        """
        Calculate the gradient of the log score for a given tree with respect to the psi and phi parameters

        :param T: Tree
        :param pos: list of pos tags
        :param m: list of tags
        :param psi: psi potentials
        :return: dlog_score/dlog_psi, dlog_score/dlog_phi
        """
        dpsi = tr.zeros_like(psi)
        for (i, j, lab) in T:
            if j != 0:
                m_i = self.get_tag_index(m[i - 1])
                pos1 = pos[i - 1]
                m_j = self.get_tag_index(m[j - 1])
                pos2 = pos[j - 1]
                dpsi[pos1, pos2, lab, m_i, m_j] += 1
        return dpsi

    def logZ(self, T, pos, psi, phi):
        """
        Belief propagation algorithm for calculating the log of the partition function Z

        :param T: tree
        :param pos: list of pos tags
        :param psi: psi potentials
        :param phi: phi potentials
        :return: log(Z) where Z is the normalizing partition function for p(m|T)
        """
        msgs = belief_propagation(T, pos, psi, phi, True)
        log_z = calculate_belief_sum(msgs, True)
        return log_z

    def dlogZ(self, T, pos, psi, phi):
        """
        Belief propagation algorithm for calculating the gradient of the log of the partition function Z

        :param T: tree
        :param pos: list of pos tags
        :param psi: psi potentials
        :param phi: phi potentials
        :return: dlog_Z/dlog_psi, dlog_Z/dlog_phi
        """
        msgs = belief_propagation(T, pos, psi, phi, True)
        dpsi = calculate_gradient(msgs, T, pos, psi, True, True)
        return dpsi

    def log_prob(self, T, pos, m, psi, phi=tr.Tensor()):
        """
        Calculate the conditional log probability of the tags given the tree

        :param T: tree
        :param pos: list of pos tags
        :param m: list of tags
        :param psi: psi potentials
        :param phi: phi potentials
        :return: log(p(m|T))
        """
        if phi.size() == tr.Size([0]):
            phi = self.create_phi(T, pos, m)
        return self.log_score(T, pos, m, psi, phi) - self.logZ(T, pos, psi, phi)

    def dlog_prob(self, T, pos, m, psi, phi=tr.Tensor()):
        """
        Calculate the gradient of the log probability for a given tree with respect to the psi and phi parameters

        :param T: tree
        :param pos: list of pos tags
        :param m: list of tags
        :param psi: psi potentials
        :param phi: phi potentials
        :return: dlog(p(m|T))/dpsi, dlog(p(m|T))/dphi
        """
        if phi.size() == tr.Size([0]):
            phi = self.create_phi(T, pos, m)
        dpsi_score = self.dlog_score(T, pos, m, psi)
        dpsi_Z = self.dlogZ(T, pos, psi, phi)
        return dpsi_score - dpsi_Z

    """
    Finding best tag sequence
    """
    def best_sequence(self, T, pos, psi, phi, fix_tags=[]):
        """
        Belief propagation (max-product) algorithm for calculating the best tag sequence for a tree

        :param T: tree
        :param pos: list of pos tags
        :param psi: psi potentials
        :param phi: phi potentials
        :param fix_tags: list of pairs of index of a word and the tag it should be fixed to
        :return: dictionary containing a tag for each node in the tree
        """
        for idx, m in fix_tags:
            phi[idx - 1, m] = 100
        # if fix_idx:
        #     phi[fix_idx - 1, fix_m] = 100
        msgs, pointers = max_product(T, pos, psi, phi, True)
        tags_dict = get_best_tags(T, msgs, pointers)
        tags = []
        for i in range(1, len(T) + 1):
            tags.append(self.get_tag(tags_dict[str(i)]))
        return tags

    """
    Unit testing functions
    """
    def logZ_brute(self, T, pos, psi, phi):
        """
        Brute force algorithm for calculating the log of the partition function Z

        :param T: tree
        :param pos: list of pos tags
        :param psi: psi potentials
        :param phi: phi potentials
        :return: log(Z) where Z is the normalizing partition function for p(m|T)
        """
        ms = self.get_all_tag_seq(len(T))
        log_scores = tr.zeros(len(ms), dtype=tr.float64)
        for i in range(len(ms)):
            log_scores[i] = self.log_score(T, pos, ms[i], psi, phi)
        log_z = logsumexp(log_scores)
        return log_z

    def best_sequence_brute(self, T, pos, psi, phi):
        """
        Brute force algorithm for calculating the log of the partition function Z

        :param T: tree
        :param pos: list of pos tags
        :param psi: psi potentials
        :param phi: phi potentials
        :return: log(Z) where Z is the normalizing partition function for p(m|T)
        """
        ms = self.get_all_tag_seq(len(T))
        log_scores = tr.zeros(len(ms), dtype=tr.float64)
        for i in range(len(ms)):
            log_scores[i] = self.log_score(T, pos, ms[i], psi, phi)
        best = ms[tr.argmax(log_scores)]
        tags = []
        for i in range(len(T)):
            tags.append(self.get_tag_index(best[i]))
        return tags

    def fd_grad(self, T, pos, psi, phi, eps=1e-5):
        """
        Finite Difference gradient computation for logZ

        :param T: tree
        :param pos: list of pos tags
        :param psi: psi potentials
        :param phi: phi potentials
        :param eps: difference parameter
        :return: dlogZ/dlog_psi, dlogZ/dlog_phi
        """
        dpsi = tr.zeros_like(psi)
        dphi = tr.zeros_like(phi)
        for pos1 in range(psi.shape[0]):
            for pos2 in range(psi.shape[1]):
                for lab in range(psi.shape[2]):
                    for i in range(psi.shape[3]):
                        for j in range(psi.shape[4]):
                            psi[pos1, pos2, lab, i, j] += eps
                            val1 = self.logZ(T, pos, psi, phi)
                            psi[pos1, pos2, lab, i, j] -= 2 * eps
                            val2 = self.logZ(T, pos, psi, phi)
                            psi[pos1, pos2, lab, i, j] += eps
                            dpsi[pos1, pos2, lab, i, j] = (val1 - val2) / (2 * eps)
        for p in range(phi.shape[0]):
            for i in range(phi.shape[1]):
                phi[p, i] += eps
                val1 = self.logZ(T, pos, psi, phi)
                phi[p, i] -= 2 * eps
                val2 = self.logZ(T, pos, psi, phi)
                phi[p, i] += eps
                dphi[p, i] = (val1 - val2) / (2 * eps)
        return dpsi, dphi
