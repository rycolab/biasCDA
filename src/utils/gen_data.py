import torch as tr
from random import choice, randint


def gen_psi(num_pos, num_labels, num_tags):
    """
    :param num_pos: number of possible pos labels
    :param num_labels: number of possible dependency labels
    :param num_tags: number of possible tags
    :return: random psi potentials
    """
    return tr.rand((num_pos, num_pos, num_labels, num_tags, num_tags), dtype=tr.float64)


def update_tree(i, j, l, T, unlabeled, labeled):
    """
    Update tree with node (i, j, l)

    :param i: node index
    :param j: head index
    :param l: label
    :param T: tree
    :param unlabeled: set of unlabeled nodes
    :param labeled: set of labeled nodes
    """
    T[i - 1] = (i, j, l)
    unlabeled.remove(i)
    labeled.add(i)


def gen_tree(n, num_labels):
    """
    :param n: number of nodes in tree
    :param num_labels: number of possible labels (including root)
    :return: Tree of size n
    """
    T = []
    # Create tree template
    for i in range(n):
        node = (i + 1, 0, 0)
        T.append(node)
    unlabeled = set([i + 1 for i in range(n)])
    labeled = set()
    # Pick root
    update_tree(choice(tuple(unlabeled)), 0, 0, T, unlabeled, labeled)
    # Populate tree
    while unlabeled:
        update_tree(choice(tuple(unlabeled)), choice(tuple(labeled)), randint(1, num_labels-1),
                    T, unlabeled, labeled)
    return T
