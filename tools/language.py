import itertools
import re
import string
from functools import partial, reduce, singledispatch
from operator import itemgetter
from typing import Callable, DefaultDict, List, Tuple

import gensim.parsing.preprocessing as gsp
import nltk
import numpy as np
from pandas.core.dtypes.inference import is_nested_list_like
from scipy.sparse import csr_matrix
import pandas as pd
from pandas.api.types import is_list_like
from fuzzywuzzy.fuzz import WRatio as weighted_ratio
from fuzzywuzzy.process import extractOne as extract_one
from IPython.core.display import Markdown, display
from pandas._typing import ArrayLike, AnyArrayLike
from unidecode import unidecode
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import casual
from os.path import normpath
from .typing import RandomSeed, SeriesOrArray, StrOrPattern
from . import utils
from ._validation import _check_if_tagged
from gensim.models.doc2vec import TaggedDocument

# Vectorized preprocessing funcs from Gensim
strip_short = np.vectorize(gsp.strip_short, otypes=[str])
strip_multiwhite = np.vectorize(gsp.strip_multiple_whitespaces, otypes=[str])
strip_numeric = np.vectorize(gsp.strip_numeric, otypes=[str])
strip_non_alphanum = np.vectorize(gsp.strip_non_alphanum, otypes=[str])
split_alphanum = np.vectorize(gsp.split_alphanum, otypes=[str])
strip_tags = np.vectorize(gsp.strip_tags, otypes=[str])
stem_text = np.vectorize(gsp.stem_text, otypes=[str])

# Vectorized preprocessing funcs from NTLK
remove_handles = np.vectorize(casual.remove_handles, otypes=[str])

# Vectorized Unicode-to-readable-ASCII converter
unidecode = np.vectorize(unidecode, otypes=[str])

TREEBANK_TAGS = frozenset(
    {
        "CC",
        "CD",
        "DT",
        "EX",
        "FW",
        "IN",
        "JJ",
        "JJR",
        "JJS",
        "LS",
        "MD",
        "NN",
        "NNP",
        "NNPS",
        "NNS",
        "PDT",
        "POS",
        "PRP",
        "PRP$",
        "RB",
        "RBR",
        "RBS",
        "RP",
        "SYM",
        "TO",
        "UH",
        "VB",
        "VBD",
        "VBG",
        "VBN",
        "VBP",
        "VBZ",
        "WDT",
        "WP",
        "WP$",
        "WRB",
    }
)


@singledispatch
def lowercase(docs: pd.Series):
    return docs.str.lower()


@lowercase.register
def _(docs: np.ndarray):
    return utils.flat_map(lambda x: x.lower(), docs)


@lowercase.register
def _(docs: list):
    return list(map(lambda x: x.lower(), docs))


def _compile_punct(punct, exclude):
    if exclude:
        punct = re.sub(fr"[{exclude}]", "", punct)
    return re.compile(fr"[{re.escape(punct)}]")


@singledispatch
def strip_punct(docs: pd.Series, repl=" ", punct=string.punctuation, exclude=""):
    re_punct = _compile_punct(punct, exclude)
    return docs.str.replace(re_punct, repl, regex=True)


@strip_punct.register
def _(docs: np.ndarray, repl=" ", punct=string.punctuation, exclude=""):
    re_punct = _compile_punct(punct, exclude)
    docs = utils.flat_map(re_punct.sub, docs, repl)
    return docs


@strip_punct.register
def _(docs: list, repl=" ", punct=string.punctuation, exclude=""):
    re_punct = _compile_punct(punct, exclude)
    return [re_punct.sub(repl, x) for x in docs]


@singledispatch
def limit_repeats(docs: pd.Series):
    return docs.map(casual.reduce_lengthening)


@limit_repeats.register
def _(docs: np.ndarray):
    return utils.flat_map(casual.reduce_lengthening, docs)


@limit_repeats.register
def _(docs: list):
    return list(map(casual.reduce_lengthening, docs))


def readable_sample(
    data: pd.Series, n: int = 10, random_state: RandomSeed = None
) -> None:
    if isinstance(data, pd.DataFrame):
        raise ValueError(f"Expected Series, got {type(data)}")
    data = data.sample(n=n, random_state=random_state)
    display(Markdown(data.to_markdown()))


def treebank_pos_to_wordnet(pos):
    """Inspired by https://stackoverflow.com/questions/15586721"""
    wordnet_pos = dict(
        J=wordnet.ADJ,
        V=wordnet.VERB,
        N=wordnet.NOUN,
        R=wordnet.ADV,
    )
    return wordnet_pos.get(pos[0].upper(), wordnet.NOUN)


def tokenize_tag(docs: pd.Series, tokenizer: Callable = None):
    if tokenizer is None:
        docs = docs.str.split()
    else:
        docs = docs.map(tokenizer)
    docs = docs.map(nltk.pos_tag)
    return docs


@singledispatch
def wordnet_lemmatize(docs: pd.Series, tokenizer: Callable = None):
    # Check if docs are POS tagged
    pretagged = _check_if_tagged(docs)

    # Tokenize and tag POS if not tagged
    if not pretagged:
        docs = tokenize_tag(docs, tokenizer=tokenizer)

    # Convert Treebank tags to Wordnet tags
    words = docs.explode().dropna()
    treebank_pos = words.map(itemgetter(1))
    wordnet_pos = treebank_pos.map(treebank_pos_to_wordnet)
    words = words.map(itemgetter(0))
    words = pd.Series(list(zip(words, wordnet_pos)), index=words.index)

    # Lemmatize
    wnl = WordNetLemmatizer()
    words = words.map(lambda x: wnl.lemmatize(*x))

    # Restore tags if docs were pretagged
    if pretagged:
        words = pd.Series(list(zip(words, treebank_pos)), index=words.index)
    return utils.implode(words).rename(docs.name).reindex_like(docs)


@wordnet_lemmatize.register
def _(docs: np.ndarray, tokenizer: Callable = None):
    shape = docs.shape
    docs = wordnet_lemmatize(pd.Series(docs.flat), tokenizer=tokenizer)
    return docs.to_numpy().reshape(shape)


def filter_pos(
    docs: pd.Series,
    include: List = None,
    exclude: List = None,
    tokenizer: Callable = None,
    as_tokens=False,
):
    if include is not None and exclude is not None:
        raise ValueError("Must pass only one: `include` or `exclude`")
    elif include is None and exclude is None:
        return docs
    elif include is None and exclude is not None:
        keep = set(TREEBANK_TAGS) - set(exclude)
    else:
        keep = set(include)
    pretagged = _check_if_tagged(docs)
    tagged = docs if pretagged else tokenize_tag(docs, tokenizer=tokenizer)
    words = utils.expand(tagged.explode().dropna(), labels=["word", "tag"])
    words = words.loc[words.tag.isin(keep), "word"]
    words = utils.implode(words)
    if not as_tokens:
        words = words.str.join(" ")
    docs = words.rename(docs.name).reindex_like(docs)
    return docs


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