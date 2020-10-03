import logging
from typing import Dict, List, Iterable
from dataset_reader.ontonotes_utils import Ontonotes, OntonotesSentence


logger = logging.getLogger(__name__)


def _normalize_word(word: str):
    if word in ("/.", "/?"):
        return word[1:]
    else:
        return word


class OntonotesNamedEntityRecognition():
    """
    This DatasetReader is designed to read in the English OntoNotes v5.0 data
    for fine-grained named entity recognition. It returns a dataset of instances with the
    following fields:

    tokens : ``TextField``
        The tokens in the sentence.
    tags : ``SequenceLabelField``
        A sequence of BIO tags for the NER classes.

    Note that the "/pt/" directory of the Onotonotes dataset representing annotations
    on the new and old testaments of the Bible are excluded, because they do not contain
    NER annotations.

    Parameters
    ----------
    domain_identifier: ``str``, (default = None)
        A string denoting a sub-domain of the Ontonotes 5.0 dataset to use. If present, only
        conll files under paths containing this domain identifier will be processed.
    coding_scheme : ``str``, (default = None).
        The coding scheme to use for the NER labels. Valid options are "BIO" or "BIOUL".

    Returns
    -------
    A ``Dataset`` of ``Instances`` for Fine-Grained NER.

    """

    def __init__(self, domain_identifier: str = None, coding_scheme: str = "BIO") -> None:
        super().__init__()
        self._domain_identifier = domain_identifier
        if domain_identifier == "pt":
            raise ValueError(
                "The Ontonotes 5.0 dataset does not contain annotations for"
                " the old and new testament sections."
            )
        self._coding_scheme = coding_scheme

    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        ontonotes_reader = Ontonotes()
        logger.info("Reading Fine-Grained NER instances from dataset files at: %s", file_path)
        if self._domain_identifier is not None:
            logger.info("Filtering to only include file paths containing the %s domain", self._domain_identifier,)

        for sentence in self._ontonotes_subset(ontonotes_reader, file_path, self._domain_identifier):
            tokens = [_normalize_word(t) for t in sentence.words]
            yield self.text_to_instance(tokens, sentence.named_entities, sentence.parse_tree)

    @staticmethod
    def _ontonotes_subset(ontonotes_reader: Ontonotes, file_path: str, domain_identifier: str) -> Iterable[OntonotesSentence]:
        """
        Iterates over the Ontonotes 5.0 dataset using an optional domain identifier.
        If the domain identifier is present, only examples which contain the domain
        identifier in the file path are yielded.
        """
        for conll_file in ontonotes_reader.dataset_path_iterator(file_path):
            if (domain_identifier is None or f"/{domain_identifier}/" in conll_file) \
                    and "/pt/" not in conll_file:
                yield from ontonotes_reader.sentence_iterator(conll_file)

    def text_to_instance(self, tokens, ner_tags=None, parse_tree=None):
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        instance_fields = {"tokens": tokens, "parse_tree": parse_tree}
        # Add "tag label" to instance
        if ner_tags is not None:
            # Uncomment the following 2 lines below if want to use BIOUL encoding scheme
            # if self._coding_scheme == "BIOUL":
            #     ner_tags = to_bioul(ner_tags, encoding="BIO")
            instance_fields["tags"] = ner_tags
        return instance_fields
