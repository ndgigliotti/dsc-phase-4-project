import itertools
import re
import string
from functools import partial, singledispatch
from operator import itemgetter
from typing import Callable, List, Sequence, Tuple, Union

import gensim.parsing.preprocessing as gsp
import nltk
import numpy as np
from numpy import ndarray
from numpy.lib.arraysetops import isin
from pandas.core.dtypes.inference import is_nested_list_like
from pandas.core.frame import DataFrame
from pandas.core.series import Series
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
from ._validation import _check_if_tagged, _check_array_1dlike
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

# Most filter functions here rely on the generic function `_process`.
# This allows them to easily handle a wide variety of input types.


@singledispatch
def _process(docs: Union[str, ListLike], func: Callable) -> Union[str, ListLike]:
    """Apply `func` to string or strings.

    Parameters
    ----------
    docs : str or list, Series, ndarray of str
        Document(s) to map `func` over. Accepts arrays of shape
        (n_samples,) or (n_samples, 1).

    func : Callable
        Callable for processing `docs`.

    Returns
    -------
    str or list, Series, ndarray of str
        Processed string(s), same type as input.
    """
    # This is the fallback dispatch
    # Try coercing novel sequence to list
    if isinstance(docs, Sequence):
        docs = _process(list(docs), func)
    else:
        raise TypeError(f"Expected str or list-like of str, got {type(docs)}")
    return docs


@_process.register
def _(docs: list, func: Callable) -> list:
    """Dispatch for list"""
    return list(map(func, docs))


@_process.register
def _(docs: Series, func: Callable) -> Series:
    """Dispatch for Series"""
    return docs.map(func)


@_process.register
def _(docs: ndarray, func: Callable) -> ndarray:
    """Dispatch for ndarray"""
    # Check that shape is (n_samples,) or (n_samples, 1)
    _check_array_1dlike(docs)

    # Map over flat array
    return utils.flat_map(func, docs)


@_process.register
def _(docs: str, func: Callable) -> str:
    """Dispatch for single string"""
    return func(docs)


def lowercase(docs: Union[str, ListLike]) -> Union[str, ListLike]:
    """Convenience function to make letters lowercase.

    Parameters
    ----------
    docs : str or list-like of str
        Document(s) to make lowercase.

    Returns
    -------
    str or list-like of str
        Lowercase document(s).
    """

    def lower(x):
        return x.lower()

    return _process(docs, lower)


def strip_short(docs: Union[str, ListLike]) -> Union[str, ListLike]:
    """Remove words of less than 3 characters.

    Thin layer over gensim.parsing.preprocessing.strip_short.

    Parameters
    ----------
    docs : str or list-like of str
        Document(s) to process.

    Returns
    -------
    str or list-like of str
        Processed document(s).
    """
    return _process(docs, gsp.strip_short)


def strip_multiwhite(docs: Union[str, ListLike]) -> Union[str, ListLike]:
    """Replace stretches of whitespace with a single space.

    Thin layer over gensim.parsing.preprocessing.strip_multiple_whitespaces.

    Parameters
    ----------
    docs : str or list-like of str
        Document(s) to process.

    Returns
    -------
    str or list-like of str
        Processed document(s).
    """
    return _process(docs, gsp.strip_multiple_whitespaces)


def strip_numeric(docs: Union[str, ListLike]) -> Union[str, ListLike]:
    """Remove numeric characters.

    Thin layer over gensim.parsing.preprocessing.strip_numeric.

    Parameters
    ----------
    docs : str or list-like of str
        Document(s) to process.

    Returns
    -------
    str or list-like of str
        Processed document(s).
    """
    return _process(docs, gsp.strip_numeric)


def strip_non_alphanum(docs: Union[str, ListLike]) -> Union[str, ListLike]:
    """Remove all non-alphanumeric characters.

    Thin layer over gensim.parsing.preprocessing.strip_non_alphanum.

    Parameters
    ----------
    docs : str or list-like of str
        Document(s) to process.

    Returns
    -------
    str or list-like of str
        Processed document(s).
    """
    return _process(docs, gsp.strip_non_alphanum)


def split_alphanum(docs: Union[str, ListLike]) -> Union[str, ListLike]:
    """Split up the letters and numerals in alphanumeric sequences.

    Thin layer over gensim.parsing.preprocessing.split_alphanum.

    Parameters
    ----------
    docs : str or list-like of str
        Document(s) to process.

    Returns
    -------
    str or list-like of str
        Processed document(s).
    """
    return _process(docs, gsp.split_alphanum)


def limit_repeats(docs: Union[str, ListLike]) -> Union[str, ListLike]:
    """Limit strings of repeating characters (e.g. 'aaaaa') to length 3.

    Thin layer over nltk.tokenize.casual.reduce_lengthening. This is
    the function used by TweetTokenizer if `reduce_len=True`.

    Parameters
    ----------
    docs : str or list-like of str
        Document(s) to process.

    Returns
    -------
    str or list-like of str
        Processed document(s).
    """
    return _process(docs, casual.reduce_lengthening)


def strip_tags(docs: Union[str, ListLike]) -> Union[str, ListLike]:
    """Remove HTML tags.

    Thin layer over gensim.parsing.preprocessing.strip_tags.

    Parameters
    ----------
    docs : str or list-like of str
        Document(s) to process.

    Returns
    -------
    str or list-like of str
        Processed document(s).
    """
    return _process(docs, gsp.strip_tags)


def stem_text(docs: Union[str, ListLike]) -> Union[str, ListLike]:
    """Apply Porter stemming to text.

    Thin layer over gensim.parsing.preprocessing.stem_text.

    Parameters
    ----------
    docs : str or list-like of str
        Document(s) to process.

    Returns
    -------
    str or list-like of str
        Processed document(s).
    """
    return _process(docs, gsp.stem_text)


def strip_handles(docs: Union[str, ListLike]) -> Union[str, ListLike]:
    """Remove Twitter @mentions or other similar handles.

    Thin layer over nltk.tokenize.casual.remove_handles. This is
    the function used by TweetTokenizer if `strip_handles=True`.

    Parameters
    ----------
    docs : str or list-like of str
        Document(s) to process.

    Returns
    -------
    str or list-like of str
        Processed document(s).
    """
    return _process(docs, casual.remove_handles)


def uni2ascii(docs: Union[str, ListLike]) -> Union[str, ListLike]:
    """Translate Unicode to ASCII in a highly readable way.

    Thin layer over unidecode.unidecode.

    Parameters
    ----------
    docs : str or list-like of str
        Document(s) to process.

    Returns
    -------
    str or list-like of str
        Processed document(s).
    """
    return _process(docs, unidecode)


def strip_punct(
    docs: Union[str, ListLike],
    repl: str = " ",
    punct: str = string.punctuation,
    exclude: str = "",
) -> Union[str, ListLike]:
    """Strip punctuation, optionally excluding some characters.

    Extension of gensim.parsing.preprocessing.strip_punctuation.

    Parameters
    ----------
    docs : str or list-like of str
        Document(s) to process.
    repl : str, optional
        Replacement character, by default " ".
    punct : str, optional
        String of punctuation symbols, by default `string.punctuation`.
    exclude : str, optional
        String of symbols to exclude, empty by default.

    Returns
    -------
    str or list-like of str
        Processed document(s).
    """
    if exclude:
        punct = re.sub(fr"[{exclude}]", "", punct)
    re_punct = re.compile(fr"[{re.escape(punct)}]")

    def sub(string):
        return re_punct.sub(repl, string)

    return _process(docs, sub)


def chain_filts(
    docs: Union[str, ListLike], filts: List[Callable]
) -> Union[str, ListLike]:
    """Apply a list of filters to docs.

    Extension of gensim.parsing.preprocessing.strip_punctuation.

    Parameters
    ----------
    docs : str or list-like of str
        Document(s) to process.
    filts : list
        List of callable filters to apply elementwise to docs.

    Returns
    -------
    str or list-like of str
        Processed document(s).
    """
    for filt in filts:
        docs = _process(docs, filt)
    return docs


def readable_sample(data: Series, n: int = 10, random_state: RandomSeed = None) -> None:
    """Display readable sample of text from `data`.

    Parameters
    ----------
    data : Series of str
        Series of strings to sample.
    n : int, optional
        Sample size, by default 10
    random_state : RandomSeed, optional
        Seed for pseudorandom number generator, by default None.
    """
    if isinstance(data, DataFrame):
        raise ValueError(f"Expected Series, got {type(data)}")
    data = data.sample(n=n, random_state=random_state)
    display(Markdown(data.to_markdown()))


def treebank2wordnet(pos: str) -> str:
    """Translate Treebank POS tag to Wordnet.

    Inspired by https://stackoverflow.com/questions/15586721

    Parameters
    ----------
    pos : str
        Treebank part of speech tag.

    Returns
    -------
    str
        Wordnet POS tag.
    """
    wordnet_pos = dict(
        J=wordnet.ADJ,
        V=wordnet.VERB,
        N=wordnet.NOUN,
        R=wordnet.ADV,
    )
    return wordnet_pos.get(pos[0].upper(), wordnet.NOUN)


def space_tokenize(docs: Union[str, ListLike]) -> Union[List[str], ListLike]:
    """Convenience function to tokenize by whitespace.

    Uses regex to split on any whitespace character.

    Parameters
    ----------
    docs : str or list-like of str
        Document(s) to tokenize.

    Returns
    -------
    list of str or list-like of lists of str
        Tokenized document(s).
    """
    re_white = re.compile(r"\s+")
    return _process(docs, re_white.split)


def tokenize_tag(
    docs: Union[str, ListLike], tokenizer: Callable = None
) -> Union[List[Tuple[str, str]], ListLike]:
    """Tokenize and POS-tag documents.

    Parameters
    ----------
    docs : str or list-like of str
        Document(s) to tokenize and tag.
    tokenizer: callable
        Callable to apply to documents. Defaults to
        `space_tokenize` if unspecified.

    Returns
    -------
    list of tuples of str or nested list-like
        Tokenized and tagged document(s).
    """
    input_type = type(docs)
    if tokenizer is None:
        docs = space_tokenize(docs)
    else:
        docs = _process(docs, tokenizer)

    # If input was single string, list of tokens
    # will confuse the `_process` dispatcher. Need
    # to handle the single string case directly.
    if input_type is str:
        docs = nltk.pos_tag(docs)
    else:
        # Feed any other input type into `_process`.
        docs = _process(docs, nltk.pos_tag)
    return docs


def tokenize_stem(
    docs: Union[str, ListLike], tokenizer: Callable = None
) -> Union[List[str], ListLike]:
    """Tokenize and Porter stem documents.

    Parameters
    ----------
    docs : str or list-like of str
        Document(s) to tokenize and stem.
    tokenizer: callable
        Callable to apply to documents. Defaults to
        `space_tokenize` if unspecified.

    Returns
    -------
    list of str or list-like of lists of str
        Tokenized document(s).
    """
    if tokenizer is None:
        docs = space_tokenize(docs)
    else:
        docs = _process(docs, tokenizer)
    docs = stem_text(docs)
    return docs


@singledispatch
def wordnet_lemmatize(
    docs: Union[str, ListLike], tokenizer: Callable = None, as_tokens: bool = False
) -> Union[str, List[str], ListLike]:
    # This is the fallback dispatch
    # Try coercing novel sequence to Series
    if isinstance(docs, Sequence):
        docs = Series(list(docs))
        docs = wordnet_lemmatize(docs, tokenizer=tokenizer, as_tokens=as_tokens)
    else:
        raise TypeError(f"Expected str or list-like of str, got {type(docs)}")
    return docs


@wordnet_lemmatize.register
def _(docs: Series, tokenizer: Callable = None, as_tokens: bool = False):
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
    return docs if as_tokens else docs.str.join(" ")


@wordnet_lemmatize.register
def _(docs: ndarray, tokenizer: Callable = None, as_tokens: bool = False):
    shape = docs.shape
    docs = wordnet_lemmatize(
        pd.Series(docs.flat), tokenizer=tokenizer, as_tokens=as_tokens
    )
    return docs.to_numpy().reshape(shape)


@wordnet_lemmatize.register
def _(docs: list, tokenizer: Callable = None, as_tokens: bool = False):
    docs = wordnet_lemmatize(pd.Series(docs), tokenizer=tokenizer, as_tokens=as_tokens)
    return docs.to_list()


@wordnet_lemmatize.register
def _(docs: str, tokenizer: Callable = None, as_tokens: bool = False):
    # Tokenize and tag POS
    words = tokenize_tag(docs)

    # Convert Treebank tags to Wordnet tags
    words, tb_pos = zip(*words)
    wn_pos = map(treebank2wordnet, tb_pos)
    words = zip(words, wn_pos)

    # Lemmatize
    wnl = WordNetLemmatizer()
    words = [wnl.lemmatize(x, y) for x, y in words]

    return words if as_tokens else " ".join(words)


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