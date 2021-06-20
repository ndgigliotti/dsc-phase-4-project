import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, Normalizer
from sklearn.utils.validation import check_is_fitted


class BaseVectorizer(TransformerMixin, _VectorizerMixin, BaseEstimator):
    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        analyzer="word",
        n_features=(2 ** 20),
        binary=False,
        norm="l2",
        dtype=np.float64,
    ):
        # associated with pre/post-processing
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.n_features = n_features
        self.ngram_range = ngram_range
        self.binary = binary
        self.norm = norm
        self.dtype = dtype

    def _validate_input(self, X):
        if isinstance(X, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )

    def build_post_pipe(self):
        post_pipe = Pipeline([("bin", None), ("norm", None)])
        if self.binary:
            post_pipe.set_params(bin=Binarizer(copy=False))
        if self.norm is not None:
            post_pipe.set_params(norm=Normalizer(norm=self.norm, copy=False))
        return post_pipe

    def analyze_docs(self, X):
        analyzer = self.build_analyzer()
        return map(analyzer, X)

    def fit(self, X, y=None):
        # triggers a parameter validation
        self._validate_input(X)
        self._warn_for_unused_params()
        self._validate_params()
        return self

    def transform(self, X):
        self._validate_input(X)
        self._validate_params()
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def _more_tags(self):
        return {"X_types": ["string"]}


class Doc2Vectorizer(BaseVectorizer):
    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        analyzer="word",
        n_features=(2 ** 20),
        binary=False,
        norm="l2",
        dtype=np.float64,
        dm_mean=None,
        dm=1,
        dbow_words=0,
        dm_concat=0,
        dm_tag_count=1,
        docvecs=None,
        docvecs_mapfile=None,
        comment=None,
        trim_rule=None,
        size=100,
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
        epochs=5,
        sorted_vocab=1,
        batch_words=10000,
    ):
        # Related to pre/post-processing
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            lowercase=lowercase,
            token_pattern=token_pattern,
            stop_words=stop_words,
            n_features=n_features,
            ngram_range=ngram_range,
            binary=binary,
            norm=norm,
            dtype=dtype,
        )

        # Related to gensim.models.Doc2Vec
        self.gensim_model = None
        self.dm_mean = dm_mean
        self.dm = dm
        self.dbow_words = dbow_words
        self.dm_concat = dm_concat
        self.dm_tag_count = dm_tag_count
        self.docvecs = docvecs
        self.docvecs_mapfile = docvecs_mapfile
        self.comment = comment
        self.trim_rule = trim_rule

        # Related to gensim.models.Word2Vec
        self.size = size
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
        self._validate_input(X)
        self._warn_for_unused_params()
        self._validate_params()

        docs = list(self.analyze_docs(X))
        tagged_docs = [TaggedDocument(words, [i]) for i, words in enumerate(docs)]
        self.gensim_model_ = Doc2Vec(
            documents=tagged_docs,
            dm_mean=self.dm_mean,
            dm=self.dm,
            dbow_words=self.dbow_words,
            dm_concat=self.dm_concat,
            dm_tag_count=self.dm_tag_count,
            docvecs=self.docvecs,
            docvecs_mapfile=self.docvecs_mapfile,
            comment=self.comment,
            trim_rule=self.trim_rule,
            vector_size=self.size,
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

    def transform(self, X):
        check_is_fitted(self)
        self._validate_input(X)
        docs = list(self.analyze_docs(X))
        vecs = [self.gensim_model_.infer_vector(doc) for doc in docs]
        vecs = np.reshape(np.array(vecs), (len(docs), self.gensim_model_.vector_size))
        post_pipe = self.build_post_pipe()
        return post_pipe.fit_transform(vecs)
