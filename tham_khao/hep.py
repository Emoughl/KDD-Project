"""HEP (Deng, 2020): mining High occupancy itemsets with Efficient Pruning strategy.

Reference-only implementation for the Hoi2Vec experiment (tham_khao/). Not part of the submission.

    O(P)   = sum over T containing P of |P| / |T|                         (Definition 2)
    UBO(P) = max_x  sum_{i>=x} n_i * l_x / l_i                            (Definition 7)
             l_1 < ... < l_u : distinct lengths of transactions containing P, n_i : their counts
    Any superset P' of P has O(P') <= UBO(P)                               (Theorem 4)

Algorithm 1 of the paper: level-wise search, occupancy-lists built by intersecting tid lists,
candidates kept only if UBO >= xi. Transactions are sorted by length (ascending) so that every
occupancy-list is also sorted by length, and items by support (ascending), as in Sect. 3.2.3.
"""
from collections import defaultdict

import numpy as np

EPS = 1e-9          # float tolerance against the threshold (e.g. 1/3 + 1/3 + 1/3 = 0.999...)


def _ubo(lengths):
    #UBO from the (ascending) lengths of the transactions in an occupancy-list."""
    l, n = np.unique(lengths, return_counts=True)
    suffix = np.cumsum((n / l)[::-1])[::-1]          # sum_{i>=x} n_i / l_i
    return float((l * suffix).max())


def mine_hoi(transactions, xi, max_len=None):
    """All itemsets P with O(P) >= xi. `xi` is an absolute occupancy value.

    transactions: list of item lists (|T| is the number of distinct items of T).
    Returns {frozenset(items): occupancy}.
    """
    trans = [list(dict.fromkeys(t)) for t in transactions]
    order = np.argsort([len(t) for t in trans], kind="stable")      # length-ascending database
    tlen = np.array([len(trans[k]) for k in order], dtype=np.float64)
    inv = defaultdict(list)
    for new_tid, k in enumerate(order):
        for i in trans[k]:
            inv[i].append(new_tid)

    # Line 1-8: 1-itemsets, items in support-ascending order
    items = sorted((i for i in inv if len(inv[i]) >= xi - EPS), key=lambda i: (len(inv[i]), i))
    result, level = {}, {}
    for i in items:
        ol = np.array(inv[i], dtype=np.int64)
        lens = tlen[ol]
        occ = float((1.0 / lens).sum())
        if occ >= xi - EPS:
            result[(i,)] = occ
        if _ubo(lens) >= xi - EPS:
            level[(i,)] = ol

    # Line 9-19: k-itemsets from two (k-1)-candidates sharing k-2 items (same prefix)
    rank = {i: r for r, i in enumerate(items)}
    k = 2
    while level and (max_len is None or k <= max_len):
        groups = defaultdict(list)
        for p in level:
            groups[p[:-1]].append(p)
        nxt = {}
        for members in groups.values():
            members.sort(key=lambda p: rank[p[-1]])
            for a in range(len(members)):
                ol_a = level[members[a]]
                for b in range(a + 1, len(members)):
                    ol = np.intersect1d(ol_a, level[members[b]], assume_unique=True)
                    if len(ol) == 0:
                        continue
                    lens = tlen[ol]
                    p = members[a] + (members[b][-1],)
                    occ = float(k * (1.0 / lens).sum())
                    if occ >= xi - EPS:
                        result[p] = occ
                    if _ubo(lens) >= xi - EPS:
                        nxt[p] = ol
        level, k = nxt, k + 1
    return {frozenset(p): o for p, o in result.items()}


def occupancy_bruteforce(transactions, max_len=None):
    #Occupancy of every itemset by enumerating all subsets (only for testing on tiny data)."""
    from itertools import combinations

    occ = defaultdict(float)
    for t in transactions:
        t = sorted(set(t))
        for r in range(1, (max_len or len(t)) + 1):
            for p in combinations(t, r):
                occ[frozenset(p)] += r / len(t)
    return dict(occ)
