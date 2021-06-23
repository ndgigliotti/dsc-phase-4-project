import itertools
import re
import string
from functools import partial, singledispatch
from operator import itemgetter
from typing import Callable, DefaultDict, List, Sequence, Tuple, Union, cast

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
from .typing import RandomSeed, ListLike, StrOrPattern
from . import utils
from ._validation import _check_if_tagged
from gensim.models.doc2vec import TaggedDocument

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
def _process(docs: list, func: Callable):
    return list(map(func, docs))


@_process.register
def _(docs: pd.Series, func: Callable):
    return docs.map(func)


@_process.register
def _(docs: np.ndarray, func: Callable):
    return utils.flat_map(func, docs)


@_process.register
def _(docs: str, func: Callable):
    return func(docs)


def lowercase(docs: Union[str, ListLike]):
    def lower(x):
        return x.lower()

    return _process(docs, lower)


def strip_short(docs: Union[str, ListLike]):
    return _process(docs, gsp.strip_short)


def strip_multiwhite(docs: Union[str, ListLike]):
    return _process(docs, gsp.strip_multiple_whitespaces)


def strip_numeric(docs: Union[str, ListLike]):
    return _process(docs, gsp.strip_numeric)


def strip_non_alphanum(docs: Union[str, ListLike]):
    return _process(docs, gsp.strip_non_alphanum)


def split_alphanum(docs: Union[str, ListLike]):
    return _process(docs, gsp.split_alphanum)


def limit_repeats(docs: Union[str, ListLike]):
    return _process(docs, casual.reduce_lengthening)


def strip_tags(docs: Union[str, ListLike]):
    return _process(docs, gsp.strip_tags)


def stem_text(docs: Union[str, ListLike]):
    return _process(docs, gsp.stem_text)


def strip_handles(docs: Union[str, ListLike]):
    return _process(docs, casual.remove_handles)


def uni2ascii(docs: Union[str, ListLike]):
    return _process(docs, unidecode)


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


@strip_punct.register
def _(docs: str, repl=" ", punct=string.punctuation, exclude=""):
    re_punct = _compile_punct(punct, exclude)
    return re_punct.sub(repl, docs)


def chain_filts(docs: Union[str, ListLike], filts: list):
    for filt in filts:
        docs = _process(docs, filt)
    return docs


def readable_sample(
    data: pd.Series, n: int = 10, random_state: RandomSeed = None
) -> None:
    if isinstance(data, pd.DataFrame):
        raise ValueError(f"Expected Series, got {type(data)}")
    data = data.sample(n=n, random_state=random_state)
    display(Markdown(data.to_markdown()))


def treebank2wordnet(pos):
    """Inspired by https://stackoverflow.com/questions/15586721"""
    wordnet_pos = dict(
        J=wordnet.ADJ,
        V=wordnet.VERB,
        N=wordnet.NOUN,
        R=wordnet.ADV,
    )
    return wordnet_pos.get(pos[0].upper(), wordnet.NOUN)


def space_tokenize(docs: Union[str, ListLike]):
    re_white = re.compile(r"\s+")
    return _process(docs, re_white.split)


def tokenize_tag(docs: Union[str, ListLike], tokenizer: Callable = None):
    if tokenizer is None:
        docs = space_tokenize(docs)
    else:
        docs = _process(docs, tokenizer)
    docs = _process(docs, nltk.pos_tag)
    return docs


@singledispatch
def wordnet_lemmatize(docs: pd.Series, tokenizer: Callable = None):
    # Tokenize and tag POS
    docs = tokenize_tag(docs, tokenizer=tokenizer)

    # Convert Treebank tags to Wordnet tags
    words = docs.explode().dropna()
    treebank_pos = words.map(itemgetter(1))
    wordnet_pos = treebank_pos.map(treebank2wordnet)
    words = words.map(itemgetter(0))
    words = pd.Series(list(zip(words, wordnet_pos)), index=words.index)

    # Lemmatize
    wnl = WordNetLemmatizer()
    words = words.map(lambda x: wnl.lemmatize(*x))

    # Rebuild exploded docs
    docs = utils.implode(words).rename(docs.name).reindex_like(docs)
    return docs.str.join(" ")


@wordnet_lemmatize.register
def _(docs: np.ndarray, tokenizer: Callable = None):
    shape = docs.shape
    docs = wordnet_lemmatize(pd.Series(docs.flat), tokenizer=tokenizer)
    return docs.to_numpy().reshape(shape)


@wordnet_lemmatize.register
def _(docs: list, tokenizer: Callable = None):
    docs = wordnet_lemmatize(pd.Series(docs), tokenizer=tokenizer)
    return docs.to_list()


@wordnet_lemmatize.register
def _(docs: str, tokenizer: Callable = None):
    # Tokenize and tag POS
    words = tokenize_tag(docs)

    # Convert Treebank tags to Wordnet tags
    words, tb_pos = zip(*words)
    wn_pos = map(treebank2wordnet, tb_pos)
    words = zip(words, wn_pos)

    # Lemmatize
    wnl = WordNetLemmatizer()
    words = [wnl.lemmatize(x, y) for x, y in words]

    return " ".join(words)


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