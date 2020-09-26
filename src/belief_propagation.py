import torch as tr
from utils.tree import *
from utils.math import logsumexp, logmatmul, maxmul, logsumexp_col, logsumexp_mat

"""
Messages are indexed by (msg_type, i, j) tuples.

msg_type is in [0, 1, 2, 3] and is equivalent to a message from a variable node to a unary factor node (V_U), from a
variable node to a binary factor node (V_B), from a unary factor node to a variable node (U_V), and from a binary
factory node to a variable node (B_V) respectively

i is source node and j is the destination node. Binary factors are marked by the dependent node
(node furthest from root)

"""

# Types of messages
V_U = 0
V_B = 1
U_V = 2
B_V = 3


def _from_var(msg_type):
    """
    :param msg_type: Type of message
    :return: True if the message is sent from a variable node (V_U, V_B, False otherwise (U_V, B_V)
    """
    if msg_type == V_U or msg_type == V_B:
        return True
    elif msg_type == U_V or msg_type == B_V:
        return False
    else:
        raise ValueError("Message type is not supported. Supported types are variables (VAR), unary factors (UNA), and"
                         "binary factors (BIN).")


def _are_children_computed(computed, T, j):
    """
    Check if all child messages of variable j have been computed.
    That is the unary factor on node j, (UNA, j, j) and binary factors between child nodes, i,
    and variable node (BIN, i, j)

    :param computed: set of computed messages, nodes are given in (is_var, i, j)
    :param T: tree
    :param j: head node
    :return: True if all child messages have been computed, False otherwise
    """
    children = get_children(T, j)
    # If variable has children, all binary factors to the head must be computed
    for i, lab in children:
        if (B_V, i, j) not in computed:
            return False
    # Unless the node is the root, the unary factor of the node must be computed
    return (U_V, j, j) in computed or j == get_root(T)


def _first_available(stack, computed, T):
    """
    Find the first available message to be computed

    :param stack: stack of messages to compute
    :param computed: set of computed messages
    :param T: tree
    :return: i, j, l, is_var where (i, j, l) is the tree edge being calculated and is_var is True if
    the message is from a variable to a factor and false otherwise
    """
    idx = 0
    while idx < len(stack):
        msg_type, i, j = stack[idx]
        # a message is available to be computed if it comes from a factor, or all its children have been computed
        if not _from_var(msg_type) or _are_children_computed(computed, T, i):
            k, lab = get_head(T, i)
            assert i == j or j == k, "Node " + str(i) + " cannot have two heads"
            stack.pop(idx)
            return i, j, lab, msg_type
        idx += 1
    raise ValueError("Stack has no possible message to compute")


def _get_msgs_to(msgs, to, from_var, exceptions=None):
    """
    Return all incoming messages of a node
    :param msgs: messages
    :param to: node index
    :param from_var: True if messages are from variables, False otherwise
    :param exceptions: list of (msg_type, node_id) of messages to ignore
    :return: incoming messages
    """
    if not exceptions:
        exceptions = []
    to_msgs = []
    for msg_type, i, j in msgs:
        if to == j and from_var == _from_var(msg_type) and (msg_type, i) not in exceptions:
            to_msgs.append(msgs[(msg_type, i, j)])
    return tr.stack(to_msgs)


def _pass_msg_var(msgs, msg_type, i, j, use_log):
    """
    Update the message from a variable i to factor j

    :param msgs: messages
    :param msg_type: type of message (is node j a unary or binary factor)
    :param i: variable node
    :param j: factor node
    :param use_log: True if operations should be done in log space, False otherwise
    """
    assert msg_type == V_U or msg_type == V_B,\
        "Message must be from a variable to a unary factor (V_U) or a binary factor (V_B)"
    ms = _get_msgs_to(msgs, i, False, [(U_V if msg_type == V_U else B_V, j)])
    assert len(ms) > 0, "Variable must have received at least one message"
    msg = tr.sum(ms, 0) if use_log else tr.prod(ms, 0)
    msgs[(msg_type, i, j)] = msg


def _pass_msg_fac(msgs, msg_type, i, j, pos1, pos2, lab, head, psi, phi, use_log, max_product, pointers):
    """
    Update the message from a factor i to a variable j
    :param msgs: messages
    :param msg_type: type of message (is node i a unary or binary factor)
    :param i: factor node
    :param j: variable node
    :param pos1: pos tag 1 of factor
    :param pos2: pos tag 2 of factor
    :param lab: label of factor
    :param psi: binary psi potentials
    :param phi: unary phi potentials
    :param use_log: True if operations should be done in log space, False otherwise
    :param max_product: True if using max-product, False otherwise (sum-product)
    :param pointers: Backpointers of best tags (only used if max_product=True
    """
    if msg_type == U_V:
        msg = phi[i - 1, :]
    elif msg_type == B_V:
        ms = _get_msgs_to(msgs, i, True, [(V_B, i if i == j else head)])
        if max_product:
            try:
                len(pointers)
            except TypeError:
                raise ValueError("Pointers must be an instantiated dictionary")
            msg, pointer = maxmul(ms[0], psi[pos1, pos2, lab, :, :], use_log)
            pointers[(msg_type, i, j)] = pointer.int()
        elif use_log:
            msg = logmatmul(ms[0], psi[pos1, pos2, lab, :, :])
        else:
            msg = tr.dot(ms[0], psi[pos1, pos2, lab, :, :])
    else:
        raise ValueError("Message must be to a variable from a unary factor (U_V) or a binary factor (B_V)")
    msgs[(msg_type, i, j)] = msg


def _pass_msgs_from_leaves(T, pos, msgs, psi, phi, use_log, max_product, pointers=None):
    """
    Message passing from leaves to root

    :param T: tree
    :param pos: list of pos tags
    :param msgs: messages
    :param psi: binary psi potentials
    :param phi: unary phi potentials
    :param use_log: True if operations should be done in log space, False otherwise
    :param max_product: True if using max-product, False otherwise (sum-product)
    :param pointers: Backpointers of best tags (only used if max_product=True
    :return: messages after message passing has been completed
    """
    computed = set()
    stack = [(U_V, i, i) for i in get_all_not_root(T)]
    root = get_root(T)
    while stack:
        i, j, lab, msg_type = _first_available(stack, computed, T)
        next_msg = None
        if _from_var(msg_type):
            _pass_msg_var(msgs, msg_type, i, i, use_log)
            if i != root:
                head, _ = get_head(T, i)
                next_msg = (B_V, i, head)
        else:
            pos_i, pos_j = pos[i - 1], pos[j - 1]
            head, _ = get_head(T, i)
            _pass_msg_fac(msgs, msg_type, i, j, pos_i, pos_j, lab, head, psi, phi, use_log, max_product, pointers)
            m_type = V_U if j == root else V_B
            next_msg = (m_type, j, j)
        if next_msg and next_msg not in stack:
            stack.append(next_msg)
        computed.add((msg_type, i, j))
    return msgs, pointers if max_product else msgs


def _pass_msgs_from_root(T, pos, msgs, psi, phi, use_log, max_product, pointers=None):
    """
    Message passing from root to leaves

    :param T: tree
    :param pos: list of pos tags
    :param msgs: messages
    :param psi: binary psi potentials
    :param phi: unary phi potentials
    :param use_log: True if operations should be done in log space, False otherwise
    :param max_product: True if using max-product, False otherwise (sum-product)
    :param pointers: Backpointers of best tags (only used if max_product=True
    :return: messages after message passing has been completed
    """
    root = get_root(T)
    stack = [(U_V, root, root)]
    while stack:
        msg_type, i, j = stack.pop(0)
        if _from_var(msg_type):
            _pass_msg_var(msgs, msg_type, i, j, use_log)
            if msg_type == V_B:
                stack.append((B_V, j, j))
        else:
            assert i == j, "All factor messages must go backwards"
            head, lab = get_head(T, i)
            pos_i = pos[i - 1]
            pos_j = pos[head - 1]
            _pass_msg_fac(msgs, msg_type, i, i, pos_j, pos_i, lab, head, psi, phi, use_log, max_product, pointers)
            if msg_type == B_V:
                stack.append((V_U, j, j))
            children = get_children(T, j)
            for k, _ in children:
                stack.append((V_B, j, k))
    return msgs, pointers


def belief_propagation(T, pos, psi, phi, use_log=True):
    """
    Belief propagation algorithm for a tree

    :param T: tree
    :param pos: list of pos tags
    :param psi: binary psi potentials
    :param phi: unary phi potentials
    :param use_log: True if operations should be done in log space, False otherwise
    :return: messages of nodes in the factor graph of the tree
    """
    # Check that tree is actually a tree
    is_tree, err = validate_tree(T)
    if not is_tree:
        raise ValueError(err)
    msgs = dict()
    # Forward step
    msgs = _pass_msgs_from_leaves(T, pos, msgs, psi, phi, use_log, False)[0]
    # Backward step, note that psi needs to be tranposed
    psi_transpose = tr.transpose(tr.transpose(psi, 0, 1), 3, 4)
    msgs = _pass_msgs_from_root(T, pos, msgs, psi_transpose, phi, use_log, False)[0]
    return msgs


def max_product(T, pos, psi, phi, use_log=True):
    """
    Max-Product algorithm for a tree

    :param T: tree
    :param pos: list of pos tags
    :param psi: binary psi potentials
    :param phi: unary phi potentials
    :param use_log: True if operations should be done in log space, False otherwise
    :return: messages of nodes in the factor graph of the tree and backpointers to best tags
    """
    # Check that tree is actually a tree
    is_tree, err = validate_tree(T)
    if not is_tree:
        raise ValueError(err)
    msgs = dict()
    pointers = dict()
    # Forward step
    msgs, pointers = _pass_msgs_from_leaves(T, pos, msgs, psi, phi, use_log, True, pointers)
    # Backward step, note that psi needs to be tranposed
    psi_transpose = tr.transpose(tr.transpose(psi, 0, 1), 3, 4)
    msgs, pointers = _pass_msgs_from_root(T, pos, msgs, psi_transpose, phi, use_log, True, pointers)
    return msgs, pointers


def marg_dist(msgs, i, use_log=True):
    """
    Calculate the marginal distribution of a variable node from the messages of a factor graph

    :param msgs: messages
    :param i:
    :param use_log: True if operations should be done in log space, False otherwise
    :return: marginal distribution of variable node i
    """
    ms = _get_msgs_to(msgs, i, False)
    if use_log:
        md = tr.sum(ms, 0)
    else:
        md = tr.prod(ms, 0)
    return md


def calculate_belief_sum(msgs, use_log=True):
    """
    Calculate the sum of the beliefs for a variable in a factor graph. This value is the same for all variable nodes

    :param msgs: messages
    :param use_log: True if operations should be done in log space, False otherwise
    :return: Sum of belief or marginals of any variable node
    """
    md = marg_dist(msgs, 1, use_log)
    if use_log:
        belief = logsumexp(md)
    else:
        belief = tr.sum(md)
    return belief


def get_best_tags(T, msgs, pointers):
    """
    Find the best tag sequence for a tree

    :param msgs: messages
    :param pointers: back pointers
    :param T: tree
    :return: dictionary containing best tag for each node in the tree
    """
    root = get_root(T)
    root_md = marg_dist(msgs, root, True)
    tags = dict()
    tags[str(root)] = tr.argmax(root_md)
    stack = [root]
    while stack:
        j = stack.pop(0)
        for i, _ in get_children(T, j):
            pointer = pointers[(B_V, i, j)]
            tags[str(i)] = pointer[tags[str(j)]]
            stack.append(i)
    return tags


def calculate_psi_margin(msgs, T, pos, pos1, pos2, lab, psi, normalize, use_log=True):
    """
    Calculate marginal distribution of one of the phi function potenials
    :param msgs: messages
    :param T: tree
    :param pos: list of pos tags
    :param pos1: pos tag 1 of factor
    :param pos2: pos tag 2 of factor
    :param lab: label of factor
    :param psi: psi potentials
    :param normalize: normalizing constant (normally belief sum)
    :param use_log: True if operations should be done in log space, False otherwise
    :return: marginal distribution of psi[idx, :, :]
    """
    ms = []
    for i, j, l in T:
        if l == 0:
            continue
        p1, p2 = pos[i - 1], pos[j - 1]
        if p1 == pos1 and p2 == pos2 and l == lab:
            mis = _get_msgs_to(msgs, i, False, [(B_V, i)])
            mjs = _get_msgs_to(msgs, j, False, [(B_V, i)])
            if use_log:
                ms.append(logmatmul(tr.sum(mis, 0).view(len(mis[0]), 1), tr.sum(mjs, 0).view(1, len(mjs[0]))))
            else:
                ms.append(tr.dot(tr.prod(mis, 0).view(len(mis[0]), 1), tr.prod(mjs, 0).view(1, len(mjs[0]))))
    ms = tr.stack(ms)
    if use_log:
        marg = psi[pos1, pos2, lab, :, :] + logsumexp_mat(ms) - normalize
    else:
        marg = psi[pos1, pos2, lab, :, :] * tr.sum(ms, 0) / normalize
    return marg


def calculate_gradient(msgs, T, pos, psi, use_log=True, take_exp=True):
    """
    Calculate the marginals of the psi and phi parameters
    :param msgs: messages
    :param T: tree
    :param pos: list of pos tags
    :param psi: psi potentials
    :param phi: phi potentials
    :param use_log: True if operations should be done in log space, False otherwise
    :param take_exp: True if marginals computed are log marginals, False otherwise
    :return: psi marginals, phi marginals
    """
    dpsi = tr.zeros_like(psi)
    psi_calculated = set()
    normalize = calculate_belief_sum(msgs, use_log)
    for i, j, lab in T:
        if lab == 0:
            continue
        pos1 = pos[i - 1]
        pos2 = pos[j - 1]
        if (pos1, pos2, lab) not in psi_calculated:
            psi_marg = calculate_psi_margin(msgs, T, pos, pos1, pos2, lab, psi, normalize, use_log)
            dpsi[pos1, pos2, lab, :, :] = tr.exp(psi_marg) if take_exp else psi_marg
            psi_calculated.add((pos1, pos2, lab))
    return dpsi
