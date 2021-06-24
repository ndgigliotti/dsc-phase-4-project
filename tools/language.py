import itertools
import re
import string
from functools import partial, singledispatch
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    List,
    Mapping,
    NoReturn,
    Sequence,
    Union,
)

import gensim.parsing.preprocessing as gsp
import nltk
import pandas as pd
from fuzzywuzzy.fuzz import WRatio as weighted_ratio
from fuzzywuzzy.process import extractOne as extract_one
from IPython.core.display import Markdown, display
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import casual
from numpy import ndarray
from pandas._typing import AnyArrayLike
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.sparse import csr_matrix
from unidecode import unidecode

from . import utils
from ._validation import _check_array_1dlike
from .typing import (
    CallableOnStr,
    Documents,
    PatternLike,
    RandomSeed,
    TaggedTokenList,
    Tokenizer,
    TokenList,
)

# Most filter functions here rely on the generic function `_process`.
# This allows them to easily handle a wide variety of input types.


@singledispatch
def _process(docs: Documents, func: CallableOnStr) -> Any:
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
    # Try coercing novel iterable to list
    if isinstance(docs, Iterable):
        docs = _process(list(docs), func)
    else:
        raise TypeError(f"Expected str or iterable of str, got {type(docs)}")
    return docs


@_process.register
def _(docs: list, func: CallableOnStr) -> list:
    """Dispatch for list"""
    return list(map(func, docs))


@_process.register
def _(docs: Series, func: CallableOnStr) -> Series:
    """Dispatch for Series"""
    return docs.map(func)


@_process.register
def _(docs: ndarray, func: CallableOnStr) -> ndarray:
    """Dispatch for ndarray"""
    # Check that shape is (n_samples,) or (n_samples, 1)
    _check_array_1dlike(docs)

    # Map over flat array
    return utils.flat_map(func, docs)


@_process.register
def _(docs: str, func: CallableOnStr) -> Any:
    """Dispatch for single string"""
    return func(docs)


def lowercase(docs: Documents) -> Documents:
    """Convenience function to make letters lowercase.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to make lowercase.

    Returns
    -------
    str or iterable of str
        Lowercase document(s).
    """

    def lower(x):
        return x.lower()

    return _process(docs, lower)


def strip_short(docs: Documents) -> Documents:
    """Remove words of less than 3 characters.

    Thin layer over gensim.parsing.preprocessing.strip_short.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    return _process(docs, gsp.strip_short)


def strip_multiwhite(docs: Documents) -> Documents:
    """Replace stretches of whitespace with a single space.

    Thin layer over gensim.parsing.preprocessing.strip_multiple_whitespaces.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    return _process(docs, gsp.strip_multiple_whitespaces)


def strip_numeric(docs: Documents) -> Documents:
    """Remove numeric characters.

    Thin layer over gensim.parsing.preprocessing.strip_numeric.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    return _process(docs, gsp.strip_numeric)


def strip_non_alphanum(docs: Documents) -> Documents:
    """Remove all non-alphanumeric characters.

    Thin layer over gensim.parsing.preprocessing.strip_non_alphanum.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    return _process(docs, gsp.strip_non_alphanum)


def split_alphanum(docs: Documents) -> Documents:
    """Split up the letters and numerals in alphanumeric sequences.

    Thin layer over gensim.parsing.preprocessing.split_alphanum.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    return _process(docs, gsp.split_alphanum)


def limit_repeats(docs: Documents) -> Documents:
    """Limit strings of repeating characters (e.g. 'aaaaa') to length 3.

    Thin layer over nltk.tokenize.casual.reduce_lengthening. This is
    the function used by TweetTokenizer if `reduce_len=True`.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    return _process(docs, casual.reduce_lengthening)


def strip_tags(docs: Documents) -> Documents:
    """Remove HTML tags.

    Thin layer over gensim.parsing.preprocessing.strip_tags.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    return _process(docs, gsp.strip_tags)


def stem_text(docs: Documents) -> Documents:
    """Apply Porter stemming to text.

    Thin layer over gensim.parsing.preprocessing.stem_text.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    return _process(docs, gsp.stem_text)


def strip_handles(docs: Documents) -> Documents:
    """Remove Twitter @mentions or other similar handles.

    Thin layer over nltk.tokenize.casual.remove_handles. This is
    the function used by TweetTokenizer if `strip_handles=True`.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    return _process(docs, casual.remove_handles)


def uni2ascii(docs: Documents) -> Documents:
    """Translate Unicode to ASCII in a highly readable way.

    Thin layer over unidecode.unidecode.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    return _process(docs, unidecode)


def strip_punct(
    docs: Documents,
    repl: str = " ",
    punct: str = string.punctuation,
    exclude: str = "",
) -> Documents:
    """Strip punctuation, optionally excluding some characters.

    Extension of gensim.parsing.preprocessing.strip_punctuation.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.
    repl : str, optional
        Replacement character, by default " ".
    punct : str, optional
        String of punctuation symbols, by default `string.punctuation`.
    exclude : str, optional
        String of symbols to exclude, empty by default.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    if exclude:
        punct = re.sub(fr"[{exclude}]", "", punct)
    re_punct = re.compile(fr"[{re.escape(punct)}]")

    def sub(string):
        return re_punct.sub(repl, string)

    return _process(docs, sub)


def chain_filts(docs: Documents, filts: Iterable[CallableOnStr]) -> Documents:
    """Apply a pipeline of filters to docs.

    Extension of gensim.parsing.preprocessing.strip_punctuation.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.
    filts : list
        List of callable filters to apply elementwise to docs.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    for filt in filts:
        docs = _process(docs, filt)
    return docs


def readable_sample(
    data: Series, n: int = 10, random_state: RandomSeed = None
) -> NoReturn:
    """Display readable sample of text from `data`.

    Parameters
    ----------
    data : Series of str
        Series of strings to sample.
    n : int, optional
        Sample size, by default 10
    random_state : int, array-like, BitGenerator or RandomState, optional
        Seed for pseudorandom number generator, by default None.
    """
    if isinstance(data, DataFrame):
        raise ValueError(f"Expected Series, got {type(data)}")
    data = data.sample(n=n, random_state=random_state)
    display(Markdown(data.to_markdown()))


def treebank2wordnet(pos: str) -> str:
    """Translate Treebank POS tag to Wordnet.

    Inspired by https://stackoverflow.com/questions/15586721.

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


def space_tokenize(docs: Documents) -> TokenList:
    """Convenience function to tokenize by whitespace.

    Uses regex to split on any whitespace character.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to tokenize.

    Returns
    -------
    list of str or iterable of lists of str
        Tokenized document(s).
    """
    re_white = re.compile(r"\s+")
    return _process(docs, re_white.split)


def tokenize_tag(
    docs: Documents,
    tokenizer: Tokenizer = space_tokenize,
) -> Union[TaggedTokenList, Collection[TaggedTokenList]]:
    """Tokenize and POS-tag documents.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to tokenize and tag.
    tokenizer: callable
        Callable to apply to documents. Defaults to
        `space_tokenize` if unspecified.

    Returns
    -------
    list of tuples of str or nested iterable
        Tokenized and tagged document(s).
    """
    input_type = type(docs)
    docs = _process(docs, tokenizer)

    # If input was single string, list of tokens
    # will confuse the `_process` dispatcher into
    # trying to tag each token individually. Need
    # to handle the singular case directly.
    if input_type is str:
        docs = nltk.pos_tag(docs)
    else:
        # Feed everything else into `_process`.
        docs = _process(docs, nltk.pos_tag)
    return docs


def tokenize_stem(
    docs: Documents, tokenizer: Tokenizer = space_tokenize
) -> Union[TokenList, Collection[TokenList]]:
    """Tokenize and Porter stem documents.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to tokenize and stem.
    tokenizer: callable
        Callable to apply to documents. Defaults to
        `space_tokenize`.

    Returns
    -------
    list of str or collection of lists of str
        Tokenized document(s).
    """
    docs = _process(docs, tokenizer)
    docs = stem_text(docs)
    return docs


@singledispatch
def wordnet_lemmatize(
    docs: Documents, tokenizer: Tokenizer = space_tokenize, as_tokens: bool = False
) -> Union[Documents, TokenList, Collection[TokenList]]:
    """[summary]

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to lemmatize.
    tokenizer : callable (str -> list of str), optional
        Callable for tokenizing document(s), by default space_tokenize.
    as_tokens : bool, optional
        Return document(s) as list(s) of tokens, by default False.

    Returns
    -------
    str, list of str (as_tokens), collection of str, collection of list of str (as_tokens)
        Lemmatized document(s), optionally as token-list(s).
    """
    # This is the fallback dispatch
    # Try coercing novel sequence to Series
    if isinstance(docs, Sequence):
        docs = Series(list(docs))
        docs = wordnet_lemmatize(docs, tokenizer=tokenizer, as_tokens=as_tokens)
    else:
        raise TypeError(f"Expected str or iterable of str, got {type(docs)}")
    return docs


@wordnet_lemmatize.register
def _(
    docs: Series, tokenizer: Tokenizer = space_tokenize, as_tokens: bool = False
) -> Series:
    """Dispatch for Series"""
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
def _(
    docs: ndarray, tokenizer: Tokenizer = space_tokenize, as_tokens: bool = False
) -> ndarray:
    """Dispatch for ndarray"""
    shape = docs.shape
    docs = wordnet_lemmatize(
        pd.Series(docs.flat), tokenizer=tokenizer, as_tokens=as_tokens
    )
    return docs.to_numpy().reshape(shape)


@wordnet_lemmatize.register
def _(
    docs: list, tokenizer: Tokenizer = space_tokenize, as_tokens: bool = False
) -> Union[TokenList, List[TokenList]]:
    """Dispatch for list"""
    docs = wordnet_lemmatize(pd.Series(docs), tokenizer=tokenizer, as_tokens=as_tokens)
    return docs.to_list()


@wordnet_lemmatize.register
def _(
    docs: str, tokenizer: Tokenizer = space_tokenize, as_tokens: bool = False
) -> Union[str, TokenList]:
    """Dispatch for str"""
    # Tokenize and tag POS
    words = tokenize_tag(docs, tokenizer=tokenizer)

    # Convert Treebank tags to Wordnet tags
    words, tb_pos = zip(*words)
    wn_pos = map(treebank2wordnet, tb_pos)
    words = zip(words, wn_pos)

    # Lemmatize
    wnl = WordNetLemmatizer()
    words = [wnl.lemmatize(x, y) for x, y in words]

    return words if as_tokens else " ".join(words)


def locate_patterns(
    *pats: PatternLike,
    strings: Series,
    exclusive: bool = False,
    flags: re.RegexFlag = 0,
) -> Series:
    """Find all occurrences of one or more regex in a string Series.

    Parameters
    ----------
    strings : Series
        Strings to find and index patterns in.
    exclusive : bool, optional
        Drop indices that match more than one pattern. False by default.
    flags : RegexFlag, optional
        Flags for regular expressions, by default 0.

    Returns
    -------
    Series
        Series of matches (str).
    """
    # Gather findings for each pattern
    findings = []
    for id_, pat in enumerate(pats):
        pat_findings = (
            strings.str.findall(pat, flags=flags)
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


@singledispatch
def fuzzy_match(
    strings: Iterable[str],
    choices: Iterable[str],
    scorer: Callable[[str, str], int] = weighted_ratio,
    **kwargs,
) -> DataFrame:
    """Fuzzy match each element of `strings` with one of `choices`.

    Parameters
    ----------
    strings : iterable of str
        Strings to find matches for.
    choices : iterable of str
        Strings to choose from.
    scorer : callable ((string, choice) -> int), optional
        Scoring function, by default weighted_ratio.

    Returns
    -------
    DataFrame
        Table of matches and scores.
    """
    # Try to coerce iterable into Series
    if isinstance(strings, ndarray):
        strings = Series(strings)
    else:
        strings = Series(list(strings))
    return fuzzy_match(strings, choices, scorer=scorer, **kwargs)


@fuzzy_match.register
def _(
    strings: Series,
    choices: Iterable[str],
    scorer: Callable[[str, str], int] = weighted_ratio,
    **kwargs,
) -> DataFrame:
    """Dispatch for Series (retains index)."""
    select_option = partial(
        extract_one,
        choices=choices,
        scorer=scorer,
        **kwargs,
    )
    scores = strings.map(select_option, "ignore")
    strings = strings.to_frame("original")
    strings["match"] = scores.map(itemgetter(0), "ignore")
    strings["score"] = scores.map(itemgetter(1), "ignore")
    return strings


def frame_doc_vecs(
    doc_vecs: csr_matrix,
    vocab: Mapping[str, int],
    doc_index: Union[List, AnyArrayLike] = None,
) -> DataFrame:
    """Convert sparse document vectors into a DataFrame with feature labels.

    Parameters
    ----------
    doc_vecs : csr_matrix
        Sparse matrix from Scikit-Learn TfidfVectorizer or similar.
    vocab : mapping (str -> int)
        Mapping from terms to feature indices.
    doc_index : list or array-like, optional
        Index for new DataFrame, defaults to a standard RangeIndex.

    Returns
    -------
    DataFrame
        Document vectors with feature labels.
    """
    vocab = utils.swap_index(Series(vocab)).sort_index()
    if doc_index is None:
        doc_index = pd.RangeIndex(0, doc_vecs.shape[0])
    return DataFrame(doc_vecs.todense(), columns=vocab.to_numpy(), index=doc_index)
