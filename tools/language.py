import itertools
import re
import string
from functools import lru_cache, partial, singledispatch, wraps
from operator import itemgetter
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    List,
    Mapping,
    NoReturn,
    Sequence,
    Type,
    Union,
)

import gensim.parsing.preprocessing as gsp
import nltk
from deprecation import deprecated
from nltk.metrics.association import NGRAM
import numpy as np
import pandas as pd
from fuzzywuzzy.fuzz import WRatio as weighted_ratio
from fuzzywuzzy.process import extractOne as extract_one
from IPython.core.display import Markdown, display
from nltk.corpus import wordnet
from nltk.sentiment.util import mark_negation as mark_neg
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import casual
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from nltk.metrics import (
    BigramAssocMeasures,
    TrigramAssocMeasures,
    QuadgramAssocMeasures,
)
from numpy import ndarray
from numpy.lib.function_base import iterable
from pandas._typing import AnyArrayLike
from pandas.core.dtypes.inference import is_list_like, is_nested_list_like
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.sparse import csr_matrix
from unidecode import unidecode

from . import utils
from ._validation import _check_1dlike, _validate_docs
from .typing import (
    CallableOnStr,
    Documents,
    PatternLike,
    SeedLike,
    TaggedTokenList,
    Tokenizer,
    TokenList,
)

NGRAM_FINDERS = MappingProxyType(
    {
        2: nltk.BigramCollocationFinder,
        3: nltk.TrigramCollocationFinder,
        4: nltk.QuadgramCollocationFinder,
    }
)
NGRAM_MEASURES = MappingProxyType(
    {
        2: nltk.BigramAssocMeasures,
        3: nltk.TrigramAssocMeasures,
        4: nltk.QuadgramAssocMeasures,
    }
)
DEFAULT_TOKENIZER = nltk.word_tokenize
DEFAULT_SEP = "<<"
CACHE_SIZE = int(5e4)
tb_tokenize = TreebankWordTokenizer().tokenize
tb_detokenize = TreebankWordDetokenizer().tokenize

# Most filter functions here rely on the generic function `_process`.
# This allows them to easily handle a wide variety of input types.


@singledispatch
def _process(docs: Documents, func: CallableOnStr, **kwargs) -> Any:
    """Apply `func` to string or strings.

    Parameters
    ----------
    docs : str or list, Series, ndarray of str
        Document(s) to map `func` over. Accepts arrays of shape
        (n_samples,) or (n_samples, 1).

    func : Callable
        Callable for processing `docs`.
    **kwargs
        Keyword arguments for `func`.

    Returns
    -------
    str or list, Series, ndarray of str
        Processed string(s), same type as input.
    """
    # This is the fallback dispatch
    # Try coercing novel iterable to list
    if isinstance(docs, Iterable):
        docs = _process(list(docs), func, **kwargs)
    else:
        raise TypeError(f"Expected str or iterable of str, got {type(docs)}")
    return docs


@_process.register
def _(docs: list, func: CallableOnStr, **kwargs) -> list:
    """Dispatch for list"""
    return list(map(partial(func, **kwargs), docs))


@_process.register
def _(docs: Series, func: CallableOnStr, **kwargs) -> Series:
    """Dispatch for Series"""
    return docs.map(partial(func, **kwargs))


@_process.register
def _(docs: ndarray, func: CallableOnStr, **kwargs) -> ndarray:
    """Dispatch for ndarray"""
    # Check that shape is (n_samples,) or (n_samples, 1)
    _check_1dlike(docs)

    # Map over flat array
    return utils.flat_map(func, docs, **kwargs)


@_process.register
def _(docs: str, func: CallableOnStr, **kwargs) -> Any:
    """Dispatch for single string"""
    return func(docs, **kwargs)


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


@singledispatch
def stem_text(docs: Documents, lowercase: bool = False) -> Documents:
    """Apply Porter stemming to text.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    # This is the dispatch for non-str types.
    if isinstance(docs, Iterable):
        docs = _process(docs, stem_text, lowercase=lowercase)
    else:
        raise TypeError(f"Expected str or iterable of str, received {type(docs)}.")
    return docs


@stem_text.register
@lru_cache(maxsize=CACHE_SIZE, typed=False)
def _(docs: str, lowercase: bool = False):
    """Dispatch for str. Keeps cache for performance."""
    stem = PorterStemmer()
    tokens = tb_tokenize(docs)
    tokens = [stem.stem(x, lowercase) for x in tokens]
    return tb_detokenize(tokens)


def strip_handles(docs: Documents) -> Documents:
    """Remove Twitter @mentions or other similar handles.

    Thin layer over nltk.tokenize.casual.remove_handles.

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


@singledispatch
def strip_stopwords(docs: Documents, stopwords: Collection[str]):
    return _process(docs, strip_stopwords, stopwords=stopwords)


@strip_stopwords.register
def _(docs: str, stopwords: Collection[str]):
    stopwords = set(stopwords)
    tokens = [x for x in tb_tokenize(docs) if x not in stopwords]
    return tb_detokenize(tokens)


@deprecated(details="Use `chain_funcs` instead.")
def chain_filts(docs: Documents, filts: List[CallableOnStr]) -> Documents:
    """Apply a pipeline of functions to docs.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.
    filts : list of callable
        Callables to apply elementwise to docs. Should take a single str argument.

    Returns
    -------
    Any
        Result of processing. Probably a str, iterable of str, or nested structure.
    """
    for filt in filts:
        docs = _process(docs, filt)

    return docs


@singledispatch
def scored_ngrams(
    docs: Documents,
    n: int = 2,
    measure: str = "pmi",
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    preprocessor: CallableOnStr = None,
    stopwords: Collection[str] = None,
    min_freq: int = 0,
    fuse_tuples: bool = True,
    sep: str = " ",
) -> Series:

    _validate_docs(docs)
    # Coerce docs to list
    if isinstance(docs, (Series, ndarray)):
        docs = docs.squeeze().tolist()
    else:
        docs = list(docs)

    # Get collocation finder and measures
    if not isinstance(n, int):
        raise TypeError(f"Expected `n` to be int, got {type(n)}.")
    if 1 < n < 5:
        n = int(n)
        finder = NGRAM_FINDERS[n]
        measures = NGRAM_MEASURES[n]()
    else:
        raise ValueError(f"Valid `n` values are 2, 3, and 4. Got {n}.")

    
    if preprocessor is not None:
        # Apply preprocessing
        docs = map(preprocessor, docs)
    if stopwords is not None:
        # Drop stopwords
        docs = strip_stopwords(docs, stopwords)

    # Find and score collocations
    ngrams = finder.from_documents(map(tokenizer, docs))
    ngrams.apply_freq_filter(min_freq)
    ngram_score = ngrams.score_ngrams(getattr(measures, measure))

    # Put the results in a DataFrame, squeeze into Series
    kind = {2: "bigram", 3: "trigram", 4: "quadgram"}[n]
    ngram_score = pd.DataFrame(ngram_score, columns=[kind, "score"])
    if fuse_tuples:
        # Join ngram tuples
        ngram_score[kind] = ngram_score[kind].str.join(sep)
    return ngram_score.set_index(kind).squeeze()


@scored_ngrams.register
def _(
    docs: str,
    n: int = 2,
    measure: str = "pmi",
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    preprocessor: CallableOnStr = None,
    stopwords: Collection[str] = None,
    min_freq: int = 0,
    fuse_tuples: bool = True,
    sep: str = " ",
) -> Series:
    # Process as singleton
    ngram_score = scored_ngrams(
        [docs],
        n=n,
        measure=measure,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        stopwords=stopwords,
        min_freq=min_freq,
        fuse_tuples=fuse_tuples,
        sep=sep,
    )
    return ngram_score


def scored_bigrams(
    docs: str,
    measure: str = "pmi",
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    preprocessor: CallableOnStr = None,
    stopwords: Collection[str] = None,
    min_freq: int = 0,
    fuse_tuples: bool = True,
    sep: str = " ",
) -> Series:
    bigram_score = scored_ngrams(
        docs,
        n=2,
        measure=measure,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        stopwords=stopwords,
        min_freq=min_freq,
        fuse_tuples=fuse_tuples,
        sep=sep,
    )
    return bigram_score


def scored_trigrams(
    docs: str,
    measure: str = "pmi",
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    preprocessor: CallableOnStr = None,
    stopwords: Collection[str] = None,
    min_freq: int = 0,
    fuse_tuples: bool = True,
    sep: str = " ",
) -> Series:
    trigram_score = scored_ngrams(
        docs,
        n=3,
        measure=measure,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        stopwords=stopwords,
        min_freq=min_freq,
        fuse_tuples=fuse_tuples,
        sep=sep,
    )
    return trigram_score


def scored_quadgrams(
    docs: str,
    measure: str = "pmi",
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    preprocessor: CallableOnStr = None,
    stopwords: Collection[str] = None,
    min_freq: int = 0,
    fuse_tuples: bool = True,
    sep: str = " ",
) -> Series:
    quadgram_score = scored_ngrams(
        docs,
        n=4,
        measure=measure,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        stopwords=stopwords,
        min_freq=min_freq,
        fuse_tuples=fuse_tuples,
        sep=sep,
    )
    return quadgram_score


def chain_funcs(docs: Documents, funcs: List[Callable]) -> Any:
    """Apply a pipeline of functions to docs.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.
    funcs : list of callable
        Callables to apply elementwise to docs. The first callable must
        take a single str argument.

    Returns
    -------
    Any
        Result of processing. Probably a str, iterable of str, or nested structure.
    """
    # Define pipeline function for singular input.
    # This allows use of `_process` for any chain of function
    # transformations which initially takes a single str argument.
    # The functions can return lists of tuples of str, or whatever,
    # as long as the first function takes a str argument.
    def process_singular(doc):
        for func in funcs:
            if not doc:
                break
            doc = func(doc)
        return doc

    # Vectorize `process_singular`
    return _process(docs, process_singular)


def make_preprocessor(funcs: List[Callable]) -> partial:
    """Create a pipeline callable which applies a chain of functions to docs.

    The resulting generic pipeline function will accept one argument
    of type str or iterable of str.

    Parameters
    ----------
    funcs : list of callable
        Callables to apply elementwise to docs. The first callable must
        take a single str argument.

    Returns
    -------
    partial object
        Generic pipeline callable.
    """
    return partial(chain_funcs, funcs=funcs)


def readable_sample(
    data: Series, n: int = 10, random_state: SeedLike = None
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


@singledispatch
def tokenize_tag(
    docs: Documents,
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    fuse_tuples: bool = False,
    sep: str = DEFAULT_SEP,
    as_tokens: bool = True,
) -> Union[
    Documents,
    TaggedTokenList,
    TokenList,
    Collection[TaggedTokenList],
    Collection[TokenList],
]:
    """Tokenize and POS-tag documents.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to tokenize and tag.
    tokenizer: callable, optional
        Callable to apply to documents.
    fuse_tuples: bool, optional
        Join tuples (word, tag) into single strings using
        separator `sep`.
    sep: str, optional
        Separator string for joining (word, tag) tuples. Only
        relevant if `fuse_tuples=True`.

    Returns
    -------
    list of tuples of str, list of str, collection of list of tuples of str,
    or collection of list of str
        Tokenized and tagged document(s).
    """
    # This is the dispatch for non-str types.
    if isinstance(docs, Iterable):
        docs = _process(
            docs,
            tokenize_tag,
            tokenizer=tokenizer,
            fuse_tuples=fuse_tuples,
            sep=sep,
            as_tokens=as_tokens,
        )
    else:
        raise TypeError(f"Expected str or iterable of str, got {type(docs)}")
    return docs


@tokenize_tag.register
@lru_cache(maxsize=CACHE_SIZE, typed=False)
def _(
    docs: str,
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    fuse_tuples: bool = False,
    sep: str = DEFAULT_SEP,
    as_tokens: bool = True,
) -> Union[str, TokenList, TaggedTokenList]:
    """Dispatch for str. Keeps cache for performance."""
    # Tuples must be fused if returning a str
    if not as_tokens:
        fuse_tuples = True

    # Tokenize and tag
    docs = tokenizer(docs)
    docs = nltk.pos_tag(docs)

    if fuse_tuples:
        # Fuse tuples
        docs = [nltk.tuple2str(x, sep) for x in docs]
    return docs if as_tokens else " ".join(docs)


@singledispatch
def mark_pos(docs: Documents, sep: str = DEFAULT_SEP):
    """Mark POS in documents.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to tokenize and tag.
    sep: str, optional
        Separator string for joining (word, tag) tuples.

    Returns
    -------
    str or collection of str
        POS marked document(s).
    """
    if isinstance(docs, Iterable):
        docs = _process(docs, mark_pos, sep=sep)
    else:
        raise TypeError(f"Expected str or iterable of str, received {type(docs)}.")
    return docs


@mark_pos.register
def _(docs: str, sep: str = DEFAULT_SEP):
    """Dispatch for str."""
    tokens = tokenize_tag(
        docs,
        tokenizer=tb_tokenize,
        fuse_tuples=True,
        sep=sep,
    )
    return tb_detokenize(tokens)


@singledispatch
def mark_negation(docs: Documents, sep: str = DEFAULT_SEP) -> Documents:
    if isinstance(docs, Iterable):
        docs = _process(docs, mark_negation, sep=sep)
    else:
        raise TypeError(f"Expected str or iterable of str, received {type(docs)}.")
    return docs


@mark_negation.register
@lru_cache(maxsize=CACHE_SIZE, typed=False)
def _(docs: str, sep: str = DEFAULT_SEP) -> str:
    """Dispatch for str. Keeps cache for performance."""
    docs = mark_neg(tb_tokenize(docs))
    for i, word in enumerate(docs):
        docs[i] = re.sub(r"_NEG$", f"{sep}NEG", word)
    return tb_detokenize(docs)


@singledispatch
def wordnet_lemmatize(docs: Documents) -> Documents:
    """Lemmatize document(s) using POS-tagging and WordNet lemmatization.

    Tags parts of speech and feeds tagged unigrams into WordNet Lemmatizer.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to lemmatize.
    tokenizer : callable (str -> list of str), optional
        Callable for tokenizing document(s).
    as_tokens : bool, optional
        Return document(s) as list(s) of tokens, by default False.

    Returns
    -------
    str, list of str (as_tokens), collection of str, collection of list of str (as_tokens)
        Lemmatized document(s), optionally as token-list(s).
    """
    # Process iterables using the str dispatch
    if isinstance(docs, Iterable):
        lemmatize = partial(wordnet_lemmatize)
        docs = _process(docs, lemmatize)
    else:
        raise TypeError(f"Expected str or iterable of str, got {type(docs)}")
    return docs


@wordnet_lemmatize.register
@lru_cache(maxsize=CACHE_SIZE, typed=False)
def _(docs: str) -> str:
    """Dispatch for str. Keeps cache for performance."""
    # Tokenize and tag POS
    words = tokenize_tag(docs, tokenizer=tb_tokenize)

    # Convert Treebank tags to Wordnet tags
    words, tb_pos = zip(*words)
    wn_pos = map(treebank2wordnet, tb_pos)
    words = zip(words, wn_pos)

    # Lemmatize
    wnl = WordNetLemmatizer()
    words = [wnl.lemmatize(x, y) for x, y in words]

    return tb_detokenize(words)


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
