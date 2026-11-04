import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROCESSED = ROOT / "notebooks" / "processed"
RESULTS_CSV = ROOT / "notebooks" / "results.csv"
FIGURES = ROOT / "notebooks" / "figures"
PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# Settings taken from the papers and from the original Trans2Vec code
DIM = 128                                         # embedding size d
EPOCHS = int(os.environ.get("KDD_EPOCHS", 50))    # Doc2Vec iterations in trans2vec_*_classify.py
N_RUNS = int(os.environ.get("KDD_N_RUNS", 10))    # each experiment is repeated 10 times
# KDD_EPOCHS / KDD_N_RUNS only exist for a quick smoke test; leave them unset for the real runs.
WORKERS = max(1, (os.cpu_count() or 2))


# ---------------------------------------------------------------------------
# 1. Reading and writing data files
# ---------------------------------------------------------------------------

def read_labeled(path):
    """Read a file in the Trans2Vec repo format: `label<TAB>tok tok tok`.

    Empty token lists are kept as empty lists (a transaction with no pattern).
    """
    docs, labels = [], []
    with open(path) as f:
        for line in f:
            label, content = line.rstrip("\n").split("\t")
            labels.append(label)
            docs.append(content.split() if content.strip() else [])
    return docs, np.array(labels)


def read_utility_db(path):
    """Read an SPMF utility file: `i1 i2 i3:TU:u1 u2 u3`.

    Returns a list of (items, utilities, transaction_utility).
    """
    out = []
    with open(path) as f:
        for line in f:
            if not line[:1].isdigit():
                continue
            items, tu, utils = line.strip().split(":")
            out.append((list(map(int, items.split())), list(map(int, utils.split())), int(tu)))
    return out


def write_utility_db(path, transactions):
    """Write a list of (items, utilities) in SPMF utility format."""
    with open(path, "w") as f:
        for items, utils in transactions:
            f.write(" ".join(map(str, items)) + ":" + str(sum(utils)) + ":" + " ".join(map(str, utils)) + "\n")


# ---------------------------------------------------------------------------
# 2. Pattern mining
# ---------------------------------------------------------------------------

def mine_fi(transactions, minsup):
    """Frequent itemsets with FP-Growth (mlxtend). Returns {frozenset: support}."""
    from mlxtend.frequent_patterns import fpgrowth
    from mlxtend.preprocessing import TransactionEncoder

    te = TransactionEncoder()
    onehot = te.fit(transactions).transform(transactions, sparse=False)
    df = pd.DataFrame(onehot, columns=te.columns_)
    fi = fpgrowth(df, min_support=minsup)
    return {frozenset(s): sup for s, sup in zip(fi["itemsets"], fi["support"])}


def mine_hui(utility_file, minutil):
    """High-utility itemsets with EFIM (PAMI implementation). Returns {frozenset: utility}."""
    import contextlib
    import io

    from PAMI.highUtilityPattern.basic import EFIM

    miner = EFIM.EFIM(str(utility_file), minutil, sep=" ")
    with contextlib.redirect_stdout(io.StringIO()):
        miner.mine()
    patterns = miner.getPatterns()
    return {frozenset(int(i) for i in k.split("\t")): int(v) for k, v in patterns.items()}


def mine_hui_maxlen2(utility_file, minutil):
    """Exact high-utility itemsets of length 1 and 2, computed with sparse matrix products.

    Used on Food, where baskets are long (up to 262 items): without a length limit, every subset
    of a single high-utility basket is itself a HUI and the number of HUIs explodes. Limiting the
    pattern length is the same idea as the length constraints of FHM+ in SPMF.
    Returns {frozenset: utility}.
    """
    import scipy.sparse as sp

    db = read_utility_db(utility_file)
    items = sorted({i for t in db for i in t[0]})
    col = {i: k for k, i in enumerate(items)}
    rows, cols, vals = [], [], []
    for t, (its, us, _) in enumerate(db):
        for i, u in zip(its, us):
            rows.append(t); cols.append(col[i]); vals.append(u)
    U = sp.csr_matrix((vals, (rows, cols)), shape=(len(db), len(items)), dtype=np.float64)
    B = (U > 0).astype(np.float64)
    single = np.asarray(U.sum(0)).ravel()
    pair = (U.T @ B).toarray()
    pair = pair + pair.T                      # u({a,b}) = sum over baskets with a and b of u(a) + u(b)
    out = {frozenset([items[k]]): int(single[k]) for k in np.where(single >= minutil)[0]}
    a_idx, b_idx = np.where(np.triu(pair, 1) >= minutil)
    for a, b in zip(a_idx, b_idx):
        out[frozenset([items[a], items[b]])] = int(pair[a, b])
    return out


def elbow(thresholds, counts, factor=2.0):
    """Elbow method used by both papers to pick minsup / minutil without labels.

    The papers describe it as: when the threshold decreases, the number of patterns increases
    slightly until a value where it increases significantly; that value is chosen. We make
    "significantly" concrete: going from high to low threshold, return the first threshold where
    the number of patterns at least doubles (factor=2). If no step doubles, return the step with
    the largest relative increase. `thresholds` and `counts` are aligned lists in any order.
    """
    pairs = sorted(zip(thresholds, counts), reverse=True)          # high -> low threshold
    ratios = [pairs[k + 1][1] / max(pairs[k][1], 1) for k in range(len(pairs) - 1)]
    for k, r in enumerate(ratios):
        if r >= factor:
            return pairs[k + 1][0]
    return pairs[int(np.argmax(ratios)) + 1][0]


def patterns_per_transaction(transactions, patterns):
    """For every transaction, list the ids of the patterns it contains (pattern is a subset)."""
    inverted = defaultdict(set)
    for t_id, items in enumerate(transactions):
        for i in items:
            inverted[i].add(t_id)
    result = [[] for _ in transactions]
    for p_id, pattern in enumerate(patterns):
        holders = set.intersection(*(inverted[i] for i in pattern))
        for t_id in holders:
            result[t_id].append(p_id)
    return result


def item_docs(transactions):
    """Items as tokens, written exactly as in the data files (same as the original code)."""
    return [[str(i) for i in t] for t in transactions]


PATTERN_PREFIX = "p"


def pattern_docs(transactions, patterns, prefix=PATTERN_PREFIX):
    """Pattern ids as tokens, numbered from 1 in order of first appearance, e.g. "p1", "p2".

    The original Trans2Vec files (food_trans_fis_*.txt, food_items_fis_*.txt) number FIs from 1
    without a prefix, so FI ids share the same numbers as item ids: in the joint model, item 509
    and FI number 509 become one token although they are unrelated. The prefix keeps the two
    vocabularies apart. Pass prefix="" to get the original (colliding) tokens.
    """
    per_t = patterns_per_transaction(transactions, patterns)
    new_id, docs = {}, []
    for ids in per_t:
        doc = []
        for p in sorted(ids):
            if p not in new_id:
                new_id[p] = len(new_id) + 1
            doc.append(f"{prefix}{new_id[p]}")
        docs.append(doc)
    return docs


def separate_pattern_tokens(docs_patterns, prefix=PATTERN_PREFIX):
    """Add the pattern prefix to pattern docs read from the original files (e.g. food_trans_fis_*)."""
    return [[prefix + tok for tok in doc] for doc in docs_patterns]


# ---------------------------------------------------------------------------
# 3. Transaction embedding (PV-DBOW)
# ---------------------------------------------------------------------------

def doc2vec(docs, seed=0, dim=DIM, epochs=EPOCHS):
    """PV-DBOW embedding, same settings as the original code (dm=0, min_count=0, 50 iterations).

    A transaction with no tokens gets a zero vector, as described in Sect. 3.4 of the Trans2Vec paper.
    """
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument

    tagged = [TaggedDocument(d, [k]) for k, d in enumerate(docs)]
    model = Doc2Vec(vector_size=dim, min_count=0, dm=0, epochs=epochs, workers=WORKERS, seed=seed)
    model.build_vocab(tagged)
    model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)
    vecs = np.array([model.dv[k] for k in range(len(docs))])
    empty = np.array([len(d) == 0 for d in docs])
    vecs[empty] = 0.0
    return vecs


def run_embedding_methods(docs_items, docs_patterns, docs_joint, y, train_idx, test_idx,
                          pattern_name="FI", family="Trans2Vec", n_runs=N_RUNS):
    """Run the four embedding methods of the paper n_runs times and return mean scores.

    TRANS-IT   : Doc2Vec on items
    TRANS-<P>  : Doc2Vec on patterns (FIs or HUIs)
    <family>-IND: average of the two vectors above
    <family>-JOI: Doc2Vec on items and patterns together
    """
    names = ["TRANS-IT", f"TRANS-{pattern_name}", f"{family}-IND", f"{family}-JOI"]
    scores = {n: [] for n in names}
    start = time.time()
    for run in range(n_runs):
        v_it = doc2vec(docs_items, seed=run)
        v_pt = doc2vec(docs_patterns, seed=run)
        v_joi = doc2vec(docs_joint, seed=run)
        for name, v in zip(names, [v_it, v_pt, (v_it + v_pt) / 2, v_joi]):
            scores[name].append(svm_score(v, y, train_idx, test_idx))
    print(f"{n_runs} lần chạy, {time.time() - start:.0f} giây")
    return summarize(scores)


# ---------------------------------------------------------------------------
# 4. Classification and logging
# ---------------------------------------------------------------------------

def svm_score(X, y, train_idx, test_idx):
    """Linear SVM with C=1 (no tuning), as in both papers. Returns (accuracy %, F1-macro %)."""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.svm import LinearSVC

    clf = LinearSVC(C=1.0, max_iter=5000)
    clf.fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])
    return 100 * accuracy_score(y[test_idx], pred), 100 * f1_score(y[test_idx], pred, average="macro")


def summarize(scores):
    """{method: [(acc, f1), ...]} -> DataFrame with mean and std."""
    rows = []
    for name, vals in scores.items():
        a = np.array(vals)
        rows.append({"method": name, "AC": a[:, 0].mean(), "AC_std": a[:, 0].std(),
                     "F1": a[:, 1].mean(), "F1_std": a[:, 1].std(), "runs": len(a)})
    return pd.DataFrame(rows).round(2)


def baselines(docs, y, train_idx, test_idx):
    """BOW, TF-IDF (paper baselines) and a length-only baseline (transaction size)."""
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    text = [" ".join(d) for d in docs]
    scores = {}
    for name, vec in [("BOW", CountVectorizer(token_pattern=r"\S+", binary=True)),
                      ("TF-IDF", TfidfVectorizer(token_pattern=r"\S+"))]:
        X = vec.fit_transform(text)
        scores[name] = [svm_score(X, y, train_idx, test_idx)]
    length = np.log1p(np.array([[len(d)] for d in docs], dtype=float))
    scores["LENGTH-ONLY"] = [svm_score(length, y, train_idx, test_idx)]
    return summarize(scores)


def log_results(df, dataset, setting=""):
    """Append a result table to results.csv (one row per method)."""
    out = df.copy()
    out.insert(0, "dataset", dataset)
    out.insert(1, "setting", setting)
    if RESULTS_CSV.exists():
        old = pd.read_csv(RESULTS_CSV)
        keep = ~((old["dataset"] == dataset) & (old["setting"] == setting) & old["method"].isin(out["method"]))
        out = pd.concat([old[keep], out], ignore_index=True)
    out.to_csv(RESULTS_CSV, index=False)
    return df


def joint_docs(docs_items, docs_patterns):
    """Joint-training docs: distinct items followed by pattern tokens.

    Same layout as the original food_items_fis_*.txt files (items first, without repeats, then patterns).
    """
    return [list(dict.fromkeys(a)) + list(b) for a, b in zip(docs_items, docs_patterns)]


# ---------------------------------------------------------------------------
# 5. Display helpers (readable tables in the notebooks)
# ---------------------------------------------------------------------------

def counts(y):
    """Label counts as a short string, e.g. 'no: 2,000 | yes: 2,000'."""
    vals, cnt = np.unique(np.asarray(y), return_counts=True)
    return " | ".join(f"{v}: {c:,}" for v, c in zip(vals, cnt))


def nice(df, caption=None, digits=2, bold_max=None, bold_min=None, formats=None):
    """Format a DataFrame for display: no index, floats rounded, optional bold max/min of a column.

    `formats` overrides the format of single columns, e.g. {"p-value": "{:.4f}"}.
    """
    df = df.reset_index(drop=True)
    fmt = {}
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            continue
        if pd.api.types.is_float_dtype(df[c]):
            fmt[c] = "{:,.%df}" % digits
        elif pd.api.types.is_integer_dtype(df[c]):
            fmt[c] = "{:,}"
    fmt.update(formats or {})
    sty = df.style.format(fmt).hide(axis="index")
    if bold_max:
        sty = sty.highlight_max(subset=[bold_max], props="font-weight: bold")
    if bold_min:
        sty = sty.highlight_min(subset=[bold_min], props="font-weight: bold")
    if caption:
        sty = sty.set_caption(caption).set_table_styles(
            [{"selector": "caption", "props": "caption-side: top; text-align: left; font-weight: bold; padding-bottom: 6px"}])
    return sty


def score_table(df, caption=None):
    """Result table from summarize(): 'mean ± std' columns, best accuracy row in bold."""
    df = df.reset_index(drop=True)

    def pm(mean, std, runs):
        return f"{mean:.2f} ± {std:.2f}" if runs > 1 else f"{mean:.2f}"

    out = pd.DataFrame({
        "Phương pháp": df["method"],
        "Accuracy (%)": [pm(a, s, r) for a, s, r in zip(df["AC"], df["AC_std"], df["runs"])],
        "F1-macro (%)": [pm(a, s, r) for a, s, r in zip(df["F1"], df["F1_std"], df["runs"])],
        "Số lần chạy": df["runs"],
    })
    best = int(np.argmax(df["AC"].to_numpy()))
    sty = out.style.hide(axis="index").apply(
        lambda row: ["font-weight: bold" if row.name == best else "" for _ in row], axis=1)
    if caption:
        sty = sty.set_caption(caption).set_table_styles(
            [{"selector": "caption", "props": "caption-side: top; text-align: left; font-weight: bold; padding-bottom: 6px"}])
    return sty
