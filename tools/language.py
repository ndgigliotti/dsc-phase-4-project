import itertools
import re
import string
from functools import partial, singledispatch
from operator import itemgetter
from typing import Callable

import gensim.parsing.preprocessing as gsp
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from fuzzywuzzy.fuzz import WRatio as weighted_ratio
from fuzzywuzzy.process import extractOne as extract_one
from IPython.core.display import Markdown, display
from pandas._typing import ArrayLike, AnyArrayLike
from unidecode import unidecode
from nltk.tokenize.casual import TweetTokenizer

from .typing import RandomBasis, SeriesOrArray, StrOrPattern
from . import utils

# Vectorized preprocessing functions from Gensim
strip_short = np.vectorize(gsp.strip_short, otypes=[str])
strip_multiwhite = np.vectorize(gsp.strip_multiple_whitespaces, otypes=[str])
strip_numeric = np.vectorize(gsp.strip_numeric, otypes=[str])
strip_non_alphanum = np.vectorize(gsp.strip_non_alphanum, otypes=[str])
split_alphanum = np.vectorize(gsp.split_alphanum, otypes=[str])
strip_tags = np.vectorize(gsp.strip_tags, otypes=[str])
stem_text = np.vectorize(gsp.stem_text, otypes=[str])

# Vectorized Unicode-to-readable-ASCII converter
unidecode = np.vectorize(unidecode, otypes=[str])


def lowercase(docs: np.ndarray):
    shape = docs.shape
    docs = np.array([x.lower() for x in docs.flat])
    return docs.reshape(shape)


def strip_punct(docs: np.ndarray, repl=" ", punct=string.punctuation, exclude=""):
    shape = docs.shape
    if exclude:
        punct = re.sub(fr"[{exclude}]", "", punct)
    re_punct = re.compile(fr"[{re.escape(punct)}]")
    docs = np.array([re_punct.sub(repl, x) for x in docs.flat])
    return docs.reshape(shape)


def tweet_tokenize(text, preserve_case=True, reduce_len=False, strip_handles=False):
    tokenizer = TweetTokenizer(
        preserve_case=preserve_case,
        reduce_len=reduce_len,
        strip_handles=strip_handles,
    )
    return tokenizer.tokenize(text)


def readable_sample(
    data: pd.Series, n: int = 10, random_state: RandomBasis = None
) -> None:
    if isinstance(data, pd.DataFrame):
        raise ValueError(f"Expected Series, got {type(data)}")
    data = data.sample(n=n, random_state=random_state)
    display(Markdown(data.to_markdown()))


def locate_patterns(
    *pats: StrOrPattern,
    docs: pd.Series,
    exclusive: bool = False,
    flags: re.RegexFlag = 0,
):
    # Gather findings for each pattern
    findings = []
    for id_, pat in enumerate(pats):
        pat_findings = (
            docs.str.findall(pat, flags=flags)
            .explode()
            .dropna()
            .to_frame("locate_patterns")
            .assign(pattern=id_)
        )
        findings.append(pat_findings)

    # Merge all findings
    findings = pd.concat(findings, axis=0)

    if exclusive:
        # Drop rows with findings from more than one pattern
        groups = findings.groupby("pattern").groups
        for key, indices in groups.items():
            groups[key] = set(indices)
        discard = set()
        for p1, p2 in itertools.combinations(groups.keys(), 2):
            discard.update(groups[p1] & groups[p2])
        findings.drop(discard, inplace=True)

    # Sort and return
    return findings.drop("pattern", axis=1).squeeze().sort_index()


def fuzzy_match(
    data: pd.Series,
    options: ArrayLike,
    scorer: Callable = weighted_ratio,
    **kwargs,
) -> pd.DataFrame:
    select_option = partial(
        extract_one,
        choices=options,
        scorer=scorer,
        **kwargs,
    )
    scores = data.map(select_option, "ignore")
    data = data.to_frame("original")
    data["match"] = scores.map(itemgetter(0), "ignore")
    data["score"] = scores.map(itemgetter(1), "ignore")
    return data


def frame_doc_vecs(
    docvecs: csr_matrix, vocab: dict, doc_index: AnyArrayLike = None
) -> pd.DataFrame:
    vocab = utils.swap_index(pd.Series(vocab)).sort_index()
    if doc_index is None:
        doc_index = pd.RangeIndex(0, docvecs.shape[0])
    return pd.DataFrame(docvecs.todense(), columns=vocab.to_numpy(), index=doc_index)