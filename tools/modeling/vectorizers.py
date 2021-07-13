import string
from functools import lru_cache, partial
from typing import Callable

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pandas.core.series import Series
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    _VectorizerMixin,
    strip_accents_ascii,
    strip_accents_unicode,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, normalize
from sklearn.utils.validation import check_is_fitted

from .. import language as lang
from .._validation import _invalid_value, _validate_raw_docs
from ..typing import CallableOnStr

# Load VADER once
vader = SentimentIntensityAnalyzer()


@lru_cache(maxsize=1_000_000, typed=False)
def _vader_polarity_scores(text: str) -> Series:
    """Return VADER polarity scores for `text`.

    Keeps cache to reuse previous results.

    Parameters
    ----------
    text : str
        Text to analyze.

    Returns
    -------
    Series
        Sentiment polarity scores.
    """
    return Series(vader.polarity_scores(text))


class VaderVectorizer(BaseEstimator, TransformerMixin):
    """Extracts VADER polarity scores from short documents.

    Parameters
    ----------
    trinarize : bool, optional
        Convert vector elements to ternary sign indicators -1.0, 0.0, and 1.0. By default False.
    category : bool, optional
        Include the positive, neutral, and negative scores in vectors, by default True.
    compound : bool, optional
        Include the compound score in vectors, by default True.
    preprocessor : CallableOnStr, optional
        Callable for preprocessing text before VADER analysis, by default None.
    norm : str, optional
        Normalization to apply, by default "l2".
    sparse : bool, optional
        Output a sparse matrix, by default True.
    """

    def __init__(
        self,
        trinarize=False,
        category=True,
        compound=True,
        preprocessor: CallableOnStr = None,
        norm=None,
        sparse=True,
    ):
        self.trinarize = trinarize
        self.category = category
        self.compound = compound
        self.preprocessor = preprocessor
        self.norm = norm
        self.sparse = sparse

    def build_postprocessor(self):
        """Construct postprocessing pipeline based on parameters."""
        pipe = Pipeline([("sign", None), ("norm", None), ("csr", None)])
        if self.trinarize:
            pipe.set_params(sign=FunctionTransformer(np.sign))
        if self.norm is not None:
            pipe.set_params(norm=Normalizer(norm=self.norm))
        if self.sparse:
            pipe.set_params(csr=FunctionTransformer(csr_matrix))
        return pipe

    def _validate_params(self):
        """Validate some parameters."""
        if not (self.category or self.compound):
            raise ValueError("Either `category` or `compound` must be True.")
        if self.preprocessor is not None:
            if not isinstance(self.preprocessor, Callable):
                raise TypeError(
                    f"Expected `preprocessor` to be callable, got {type(self.preprocessor)}"
                )

    def get_feature_names(self):
        """Return list of feature names."""
        return self.feature_names_

    def fit(self, X, y=None):
        """Does nothing except validate parameters and save feature names."""
        self._validate_params()
        _validate_raw_docs(X)
        self.feature_names_ = []
        if self.category:
            self.feature_names_ += ["neg", "neu", "pos"]
        if self.compound:
            self.feature_names_.append("comp")
        return self

    def transform(self, X):
        """Extracts the polarity scores and applies postprocessing."""
        # Input and param validation
        self._validate_params()
        _validate_raw_docs(X)

        # Apply preprocessing
        docs = X
        if self.preprocessor is not None:
            docs = self.preprocessor(docs)

        # Perform VADER analysis
        vecs = pd.DataFrame([_vader_polarity_scores(x) for x in docs])
        if self.compound and not self.category:
            vecs = vecs.loc[:, ["comp"]]
        if self.category and not self.compound:
            vecs = vecs.loc[:, ["neg", "neu", "pos"]]
        self.feature_names_ = vecs.columns.to_list()

        # Apply postprocessing and return
        postprocessor = self.build_postprocessor()
        return postprocessor.fit_transform(vecs.to_numpy())


class VectorizerMixin(_VectorizerMixin):
    def build_preprocessor(self):
        if self.preprocessor is not None:
            return self.preprocessor

        pipe = []

        # Make case insensitive
        if self.lowercase:
            pipe.append(lang.lowercase)

        # Strip accents
        if not self.strip_accents:
            pass
        elif callable(self.strip_accents):
            pipe.append(self.strip_accents)
        elif self.strip_accents == "ascii":
            pipe.append(strip_accents_ascii)
        elif self.strip_accents == "unicode":
            pipe.append(strip_accents_unicode)
        else:
            _invalid_value("strip_accents", self.strip_accents)

        # Strip HTML tags
        if self.strip_html:
            pipe.append(lang.strip_html)

        # Strip numerals
        if self.strip_numeric:
            pipe.append(lang.strip_numeric)

        # Strip Twitter @handles
        if self.strip_twitter_handles:
            pipe.append(lang.strip_twitter_handles)

        # Split alphanumeric, i.e. 'paris64' -> 'paris 64'
        if self.split_alphanum:
            pipe.append(lang.split_alphanum)

        # Strip punctuation
        if self.strip_punct:
            if isinstance(self.strip_punct, str):
                pipe.append(partial(lang.strip_punct, punct=self.strip_punct))
            else:
                pipe.append(lang.strip_punct)

        # Strip all non-alphanumeric
        if self.alphanum_only:
            pipe.append(lang.strip_non_alphanum)

        # Strip extra whitespaces
        if self.strip_multiwhite:
            pipe.append(lang.strip_multiwhite)

        # Wrap `pipe` into single callable
        return lang.make_preprocessor(pipe)

    def build_tokenizer(self):
        # Start pipeline with tokenizer
        tokenizer = super().build_tokenizer()
        pipe = [tokenizer]

        # Filter tokens by length
        min_char, max_char = self.filter_length
        if min_char or max_char:
            pipe.append(
                partial(lang.filter_length, min_char=min_char, max_char=max_char)
            )

        # Stem or lemmatize
        if not self.stemmer:
            pass
        elif callable(self.stemmer):
            pipe.append(self.stemmer)
        elif self.stemmer == "porter":
            pipe.append(lang.porter_stem)
        elif self.stemmer == "wordnet":
            pipe.append(lang.wordnet_lemmatize)
        else:
            _invalid_value("stemmer", self.stemmer)

        # Mark POS, Negation, or other
        if not self.mark:
            pass
        elif callable(self.mark):
            pipe.append(self.mark)
        elif self.mark == "neg":
            pipe.append(lang.mark_negation)
        elif self.mark == "neg_split":
            pipe.append(partial(lang.mark_negation, split=True))
        elif self.mark == "speech":
            pipe.append(partial(lang.pos_tag, fuse_tuples=True))
        elif self.mark == "speech_split":
            pipe.append(partial(lang.pos_tag, split_tuples=True))
        elif self.mark == "speech_replace":
            pipe.append(partial(lang.pos_tag, replace=True))
        else:
            _invalid_value("mark", self.mark)

        # Wrap `pipe` into single callable
        return lang.make_preprocessor(pipe)

    def get_stop_words(self):
        """Build or fetch the effective stop words set.

        Returns
        -------
        stop_words: frozenset or None
                A set of stop words.
        """
        # Do nothing if None
        if self.stop_words is None:
            result = None
        # Process string input
        elif isinstance(self.stop_words, str):
            result = lang.fetch_stopwords(self.stop_words)
        # Assume collection if not str or None
        else:
            result = frozenset(self.stop_words)
        return result

    def _validate_params(self):
        super()._validate_params()
        # Check `input`
        valid_input = {"filename", "file", "content"}
        if self.input not in valid_input:
            _invalid_value("input", self.input, valid_input)
        # Check `decode_error`
        valid_decode = {"strict", "ignore", "replace"}
        if self.decode_error not in valid_decode:
            _invalid_value("decode_error", self.decode_error, valid_decode)
        # Check `strip_accents`
        valid_accent = {"ascii", "unicode", None}
        if self.strip_accents not in valid_accent:
            if not callable(self.strip_accents):
                _invalid_value("strip_accents", self.strip_accents, valid_accent)
        # Check `strip_punct`
        if isinstance(self.strip_punct, str):
            if not set(self.strip_punct).issubset(string.punctuation):
                _invalid_value(
                    "strip_punct", self.strip_punct, f"subset of '{string.punctuation}'"
                )
        # Check `filter_length`
        if len(self.filter_length) != 2:
            _invalid_value("filter_length", self.filter_length)
        min_char, max_char = self.filter_length
        if (min_char and max_char) and min_char > max_char:
            _invalid_value("filter_length", self.filter_length)
        # Check `stemmer`
        valid_stemmer = {"porter", "wordnet", None}
        if self.stemmer not in valid_stemmer:
            if not callable(self.stemmer):
                _invalid_value("stemmer", self.stemmer, valid_stemmer)
        # Check `mark`
        valid_mark = {
            "neg",
            "neg_split",
            "neg_replace",
            "speech",
            "speech_split",
            "speech_replace",
            None,
        }
        if self.mark not in valid_mark:
            _invalid_value("mark", self.mark, valid_mark)


class FreqVectorizer(TfidfVectorizer, VectorizerMixin):
    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        strip_multiwhite=False,
        strip_numeric=False,
        split_alphanum=False,
        alphanum_only=False,
        strip_punct=False,
        strip_twitter_handles=False,
        strip_html=False,
        limit_repeats=False,
        filter_length=(None, None),
        stemmer=None,
        mark=None,
        preprocessor=None,
        tokenizer=None,
        analyzer="word",
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.float64,
        norm=None,
        use_idf=False,
        smooth_idf=True,
        sublinear_tf=False,
    ):
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
        )

        self.strip_multiwhite = strip_multiwhite
        self.strip_numeric = strip_numeric
        self.split_alphanum = split_alphanum
        self.alphanum_only = alphanum_only
        self.strip_punct = strip_punct
        self.strip_twitter_handles = strip_twitter_handles
        self.strip_html = strip_html
        self.limit_repeats = limit_repeats
        self.filter_length = filter_length
        self.stemmer = stemmer
        self.mark = mark


class Doc2Vectorizer(BaseEstimator, VectorizerMixin, TransformerMixin):
    """Doc2Vec Vectorizer for Scikit Learn API with the standard preprocessing.

    Largely derived from gensim.sklearn_api.D2VTransformer. This class exists
    because the Gensim transformer is no longer being maintained and lacks
    preprocessing functionality.

    """

    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        strip_multiwhite=True,
        strip_numeric=False,
        split_alphanum=False,
        alphanum_only=False,
        strip_punct=False,
        strip_twitter_handles=False,
        strip_html=False,
        limit_repeats=False,
        filter_length=(None, None),
        stemmer=None,
        mark=None,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        analyzer="word",
        norm="l2",
        dm_mean=None,
        dm=1,
        dbow_words=0,
        dm_concat=0,
        dm_tag_count=1,
        comment=None,
        trim_rule=None,
        vector_size=100,
        alpha=0.025,
        window=5,
        min_count=5,
        max_vocab_size=None,
        sample=1e-3,
        seed=1,
        workers=3,
        min_alpha=0.0001,
        hs=0,
        negative=5,
        cbow_mean=1,
        hashfxn=hash,
        epochs=10,
        sorted_vocab=1,
        batch_words=10000,
    ):
        # Related to pre/post-processing
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.strip_multiwhite = strip_multiwhite
        self.strip_numeric = strip_numeric
        self.split_alphanum = split_alphanum
        self.alphanum_only = alphanum_only
        self.strip_punct = strip_punct
        self.strip_twitter_handles = strip_twitter_handles
        self.strip_html = strip_html
        self.limit_repeats = limit_repeats
        self.filter_length = filter_length
        self.stemmer = stemmer
        self.mark = mark
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.norm = norm

        # Related to gensim.models.Doc2Vec
        self.dm_mean = dm_mean
        self.dm = dm
        self.dbow_words = dbow_words
        self.dm_concat = dm_concat
        self.dm_tag_count = dm_tag_count
        self.comment = comment
        self.trim_rule = trim_rule

        # Related to gensim.models.Word2Vec
        self.vector_size = vector_size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.hashfxn = hashfxn
        self.epochs = epochs
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words

    def fit(self, X, y=None):
        # Parameter validation
        _validate_raw_docs(X)
        self._warn_for_unused_params()
        self._validate_params()
        analyzer = self.build_analyzer()
        docs = [analyzer(doc) for doc in X]
        tagged_docs = [TaggedDocument(words, [i]) for i, words in enumerate(docs)]
        self.d2v_model_ = Doc2Vec(
            documents=tagged_docs,
            dm_mean=self.dm_mean,
            dm=self.dm,
            dbow_words=self.dbow_words,
            dm_concat=self.dm_concat,
            dm_tag_count=self.dm_tag_count,
            comment=self.comment,
            trim_rule=self.trim_rule,
            vector_size=self.vector_size,
            alpha=self.alpha,
            window=self.window,
            min_count=self.min_count,
            max_vocab_size=self.max_vocab_size,
            sample=self.sample,
            seed=self.seed,
            workers=self.workers,
            min_alpha=self.min_alpha,
            hs=self.hs,
            negative=self.negative,
            cbow_mean=self.cbow_mean,
            hashfxn=self.hashfxn,
            epochs=self.epochs,
            sorted_vocab=self.sorted_vocab,
            batch_words=self.batch_words,
        )
        return self

    def get_vectors(self):
        check_is_fitted(self, "d2v_model_")
        # vecs = [self.d2v_model_.dv[i] for i in self.d2v_model_.dv.index_to_key]
        vecs = self.d2v_model_.dv.vectors
        if self.norm is not None:
            vecs = normalize(vecs, norm=self.norm, copy=False)
        return vecs

    def transform(self, X):
        check_is_fitted(self, "d2v_model_")
        _validate_raw_docs(X)
        analyzer = self.build_analyzer()
        docs = [analyzer(doc) for doc in X]
        vecs = [self.d2v_model_.infer_vector(doc) for doc in docs]
        vecs = np.reshape(np.array(vecs), (len(docs), self.d2v_model_.vector_size))
        if self.norm is not None:
            vecs = normalize(vecs, norm=self.norm, copy=False)
        return vecs

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).get_vectors()
