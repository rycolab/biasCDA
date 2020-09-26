"""
A tree T is represented by a list of ordered tuples (i, j, l) such that there exists a dependency from
node i to node j with label l.

Trees are given in order of i (i.e. [(0, j_0, l_0), (1, j_1, l_1), ...] and -1 marks the imaginary root node.

It is assumed label l=0 is the 'root' dependency
"""


def get_root(T):
    """
    :param T: tree
    :return: root node of the tree
    """
    root_found = False
    root = 1
    for (i, j, l) in T:
        if j == 0:
            if root_found:
                raise ValueError("Tree cannot have more than one root!")
            else:
                root = i
                root_found = True
    if not root_found:
        raise ValueError("Tree does not have a root! All trees must have a root")
    else:
        return root


def get_leaves(T):
    """
    :param T: tree
    :return: leaf tuples of the tree
    """
    leaves = set(T)
    for (i, j, l) in T:
        if j == 0:
            continue
        head, head_l = get_head(T, j)
        leaves.discard((j, head, head_l))
    return leaves


def get_children(T, j):
    """
    :param T: tree
    :param j: node
    :return: list of child-node, label tuples of children
    """
    labels = []
    for (i, k, l) in T:
        if j == k:
            labels.append((i, l))
    return labels


def get_head(T, i):
    """
    :param T: tree
    :param i: node
    :return: head node and label of i
    """
    if 1 > i or i > len(T):
        raise ValueError(str(i) + " must be an index of the tree (0 < i <= " + str(len(T)) + ")")
    for (k, j, l) in T:
        if i == k:
            return j, l
    raise ValueError("The tree must have a head for every valid node")


def get_all_not_root(T):
    """
    :param T: tree
    :return: return all node indices that are not the root
    """
    nodes = [i + 1 for i in range(len(T))]
    nodes.pop(get_root(T) - 1)
    return nodes


def _check_well_connected(T):
    """
    Check that a tree does not have any cycles or disconnections

    :param T: tree
    :return: True if T does not have any cycles or disconnections, False otherwise
    """
    root = get_root(T)
    unreached = set([i + 1 for i in range(len(T))])
    unreached.remove(root)
    reached = set([root])
    stack = [root]
    while stack:
        current = stack.pop(-1)
        for i, _ in get_children(T, current):
            if i in reached:
                return False, "Tree contains cycles"
            reached.add(i)
            unreached.remove(i)
            stack.append(i)
    if unreached:
        return False, "Tree contains at least one unconnected node"
    else:
        return True, ""


def _is_misindexed(T):
    """
    :param T: tree
    :return: True if indices of T are not in the write order and not in [0, len(T) ), False otherwise
    """
    for x in range(1, len(T) + 1):
        i, _, _ = T[x -1]
        if not i == x:
            return True
    return False


def validate_tree(T):
    """
    :param T: tree
    :return: True if T is indeed a tree, False otherwise
    """
    try:
        get_root(T)
    except ValueError as e:
        return False, str(e)
    if _is_misindexed(T):
        return False, "Tree does not contain the right indexes or indexes are not in the right order"
    return _check_well_connected(T)


def label_used(T, lab):
    """
    :param T: tree
    :param lab: label
    :return: list of edges with label name
    """
    used = []
    for i, j, l in T:
        if l == lab:
            used.append((i, j))
    return used
