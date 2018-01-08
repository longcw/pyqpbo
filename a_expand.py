import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

from .pyqpbo import QPBO_wrapper


def binary_expand(affinity_matrix, labels, li):
    new_labels = np.copy(labels)

    non_li = labels != li
    n = np.sum(non_li)
    if n == 0:
        return new_labels

    # unary term takes into account the li nieghbors
    u_term = np.zeros([n, 2], dtype=np.float32)
    u_term[:, 0] = np.sum(affinity_matrix[non_li][:, ~non_li], 1).reshape(-1)

    # only non li label nodes participating
    rlabels = labels[non_li]
    jj, ii = np.where(affinity_matrix[non_li][:, non_li])
    wij = affinity_matrix[jj, ii]

    # select lower tri of w
    lt = ii > jj
    ii = ii[lt]
    jj = jj[lt]
    wij = wij[lt]

    E = ii.size

    p_term = np.zeros([E, 6], dtype=np.float32)
    p_term[:, 0] = ii
    p_term[:, 1] = jj
    p_term[:, 2] = wij * np.not_equal(rlabels[ii], rlabels[jj])
    p_term[:, 3] = wij
    p_term[:, 4] = wij

    ig = np.zeros(n, dtype=np.int32)
    X = QPBO_wrapper(u_term, p_term, ig)
    X[X < 0] = ig[X < 0]

    rlabels[X == 1] = li
    new_labels[non_li] = rlabels

    return new_labels


def cc_energy(affinity_matrix, labels):

    N = affinity_matrix.shape[0]

    # shape [N, N]
    # is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    ll = np.tile(labels, (N, 1))
    is_pos = ll == ll.transpose()
    energy = -np.sum(affinity_matrix[is_pos & (~np.eye(N, dtype=np.bool))])
    return energy


def a_expand(affinity_matrix):
    n = affinity_matrix.shape[0]
    if n == 0:
        return []
    affinity_matrix -= np.diag(np.diag(affinity_matrix))

    labels = np.ones(n, dtype=np.int32)
    NL = 1

    cE = cc_energy(affinity_matrix, labels)
    while True:
        accepted = False

        li = 0
        while li <= NL:
            li += 1

            new_labels = binary_expand(affinity_matrix, labels, li)
            nE = cc_energy(affinity_matrix, new_labels)
            print(new_labels, nE)

            if nE < cE:
                cE = nE
                labels = new_labels
                accepted = True
                NL = np.max(new_labels)

        if not accepted:
            break
    return labels


def influence_map(cost_matrix):
    if cost_matrix.size == 0:
        return np.ones(cost_matrix.shape[0] + cost_matrix.shape[1], dtype=np.int)
    cost_matrix = cost_matrix / 100000.
    matches = linear_assignment(cost_matrix)
    match_inds = list(zip(*matches))
    max_cost = np.max(cost_matrix[match_inds])
    soft_cost = 0.01

    affinity = -cost_matrix + max_cost + soft_cost
    affinity[match_inds] = 1
    sc_costMatrix = np.vstack([np.hstack([0 * np.eye(affinity.shape[0]), affinity]),
                               np.hstack([affinity.transpose(), 0 * np.eye(affinity.shape[1])])])

    labels = a_expand(sc_costMatrix)
    return labels


if __name__ == '__main__':
    def test():
        costMatrix = np.asarray([
            [0.0095, 0.3219, 0.2934],
            [0.3226, 0.0090, 0.3808],
            [0.2925, 0.3767, 0.0074]
        ])
        maxcost = 0.0095
        softcost = 0.05
        # affinity = -costMatrix + maxcost + softcost
        # affinity = -costMatrix + 0.05

        affinity = np.asarray([
            [1.0000, -0.2624, -0.2339],
            [-0.2630, 1.0000, -0.3213],
            [-0.2329, -0.3172, 1.0000],
        ])

        sc_costMatrix = np.vstack([np.hstack([0 * np.eye(affinity.shape[0]), affinity]),
                                   np.hstack([affinity.transpose(), 0 * np.eye(affinity.shape[1])])])

        C = a_expand(sc_costMatrix)

        print(C)

    test()
