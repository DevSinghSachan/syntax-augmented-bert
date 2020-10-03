from typing import DefaultDict, List, Optional, Iterator, Set, Tuple
from collections import defaultdict
import os
import logging


logger = logging.getLogger(__name__)

TypedSpan = Tuple[int, Tuple[int, int]]
TypedStringSpan = Tuple[str, Tuple[int, int]]


class CoNLL2005Sentence:
    """
    A class representing the annotations available for a single CONLL formatted sentence.
    Parameters
    ----------
    domain_placeholder : ``str``
        This is a variation on the document filename
    sentence_id : ``int``
        The integer ID of the sentence within a document.
    words : ``List[str]``
        This is the tokens as segmented/tokenized in the Treebank.
    pos_tags : ``List[str]``
        This is the Penn-Treebank-style part of speech. When parse information is missing,
        all parts of speech except the one for which there is some sense or proposition
        annotation are marked with a XX tag. The verb is marked with just a VERB tag.
    dependency_parse_head: ``List[int]``
    dependency_parse_label: ``List[str]``
    predicate_lemmas : ``List[Optional[str]]``
        The predicate lemma of the words for which we have semantic role
        information or word sense information. All other indices are ``None``.
    predicate_framenet_ids : ``List[Optional[int]]``
        The PropBank frameset ID of the lemmas in ``predicate_lemmas``, or ``None``.
    named_entities : ``List[str]``
        The BIO tags for named entities in the sentence.
    srl_frames : ``List[Tuple[str, List[str]]]``
        A dictionary keyed by the verb in the sentence for the given
        Propbank frame labels, in a BIO format.
    """

    def __init__(
        self,
        document_id: str,
        sentence_id: int,
        words: List[str],
        pos_tags: List[str],
        dependency_parse_head: List[int],
        dependency_parse_label: List[str],
        predicate_lemmas: List[Optional[str]],
        named_entities: List[str],
        srl_frames: List[Tuple[str, List[str]]]
    ) -> None:

        self.document_id = document_id
        self.sentence_id = sentence_id
        self.words = words
        self.pos_tags = pos_tags
        self.dependency_parse_head = dependency_parse_head
        self.dependency_parse_label = dependency_parse_label
        self.predicate_lemmas = predicate_lemmas
        self.named_entities = named_entities
        self.srl_frames = srl_frames


class CoNLL2005:
    """
    This DatasetReader is designed to read in the CoNLL-2005 SRL data.
    The file path provided to this class can then be any of the train, test or development.
    The data has the following format, ordered by column.
    1 Domain Placeholder : ``str``
    2 Sentence ID : ``int``
    3 Word ID : ``int``
        This is the word index of the word in that sentence.
    4 Word : ``str``
        This is the token as segmented/tokenized in the Treebank.
    5 Gold POS Tag : ``str``
    6 Predicted POS Tag : ``str``
        This is the POS tag predicted from the StanfordNLP POS tagger
    7 Dependency Parse Head: ``int``
        This is the dependency parse head ID of the word. We follow the StanfordCoreNLP notation for these IDs.
    8 Dependency Parse Relation: ``int``
        This is the dependency parse relation of the word. We follow the StanfordCoreNLP notation for these IDs.
    9 Placeholder:
    10 Verb Sense:
    11 Predicate lemma: ``str``
        The predicate lemma is mentioned for the rows for which we have semantic role
        information or word sense information. All other rows are marked with a "-".
    12 Placeholder:
    13 Placeholder:
    14 Named Entities: ``str``
        These columns identifies the spans representing various named entities. For documents
        which do not have named entity annotation, each line is represented with an ``*``.
    15+ Predicate Arguments: ``str``
        There is one column each of predicate argument structure information for the predicate
        mentioned in Column 7. If there are no predicates tagged in a sentence this is a
        single column with all rows marked with an ``*``.
    """

    def dataset_iterator(self, file_path: str) -> Iterator[CoNLL2005Sentence]:
        """
        An iterator over the entire dataset, yielding all sentences processed.
        """
        for conll_file in self.dataset_path_iterator(file_path):
            yield from self.sentence_iterator(conll_file)

    @staticmethod
    def dataset_path_iterator(file_path: str) -> Iterator[str]:
        """
        An iterator returning file_paths in a directory
        containing CONLL-formatted files.
        """
        logger.info("Reading CONLL sentences from dataset files at: %s", file_path)
        for root, _, files in list(os.walk(file_path)):
            for data_file in files:
                # These are a relic of the dataset pre-processing. Every
                # file will be duplicated - one file called filename.gold_skel
                # and one generated from the preprocessing called filename.gold_conll.
                if not data_file.endswith("gold_conll"):
                    continue

                yield os.path.join(root, data_file)

    def dataset_document_iterator(self, file_path: str) -> List[CoNLL2005Sentence]:
        """
        An iterator over CONLL formatted files which yields documents, regardless
        of the number of document annotations in a particular file. This is useful
        for conll data which has been preprocessed, such as the preprocessing which
        takes place for the 2012 CONLL Coreference Resolution task.
        """
        with open(file_path) as open_file:
            conll_rows = []
            document: List[CoNLL2005Sentence] = []
            for line in open_file:
                line = line.strip()
                if line != "" and not line.startswith("#"):
                    # Non-empty line. Collect the annotation.
                    conll_rows.append(line)
                else:
                    if conll_rows:
                        document.append(self._conll_rows_to_sentence(conll_rows))
                        conll_rows = []
            if document:
                return document

    def sentence_iterator(self, file_path: str) -> Iterator[CoNLL2005Sentence]:
        """
        An iterator over the sentences in an individual CONLL formatted file.
        """
        for sentence in self.dataset_document_iterator(file_path):
            yield sentence

    def _conll_rows_to_sentence(self, conll_rows: List[str]) -> CoNLL2005Sentence:
        document_id: str = None
        sentence_id: int = None
        # The words in the sentence.
        sentence: List[str] = []
        # The pos tags of the words in the sentence.
        pos_tags: List[str] = []
        # the pieces of the parse tree.
        dep_parse_head: List[int] = []
        dep_parse_rel: List[str] = []
        # The lemmatised form of the words in the sentence which
        # have SRL or word sense information.
        predicate_lemmas: List[str] = []

        verbal_predicates: List[str] = []
        span_labels: List[List[str]] = []
        current_span_labels: List[str] = []

        for index, row in enumerate(conll_rows):
            conll_components = row.split()
            document_id = conll_components[0]
            sentence_id = int(conll_components[1])
            word = conll_components[3]
            pos_tag = conll_components[4]
            dependency_parse_head = int(conll_components[6])
            dependency_parse_relation = conll_components[7]
            lemmatised_word = conll_components[10]

            if not span_labels:
                # If this is the first word in the sentence, create
                # empty lists to collect the NER and SRL BIO labels.
                # We can't do this upfront, because we don't know how many
                # components we are collecting, as a sentence can have
                # variable numbers of SRL frames.
                span_labels = [[] for _ in conll_components[13:]]
                # Create variables representing the current label for each label
                # sequence we are collecting.
                current_span_labels = [None for _ in conll_components[13:]]

            self._process_span_annotations_for_word(
                conll_components[13:], span_labels, current_span_labels
            )

            # If any annotation marks this word as a verb predicate,
            # we need to record its index. This also has the side effect
            # of ordering the verbal predicates by their location in the
            # sentence, automatically aligning them with the annotations.
            word_is_verbal_predicate = any(["(V" in x for x in conll_components[13:]])
            if word_is_verbal_predicate:
                verbal_predicates.append(word)

            sentence.append(word)
            pos_tags.append(pos_tag)
            dep_parse_head.append(dependency_parse_head)
            dep_parse_rel.append(dependency_parse_relation)
            predicate_lemmas.append(lemmatised_word if lemmatised_word != "-" else None)

        named_entities = span_labels[0]
        srl_frames = [
            (predicate, labels) for predicate, labels in zip(verbal_predicates, span_labels[1:])
        ]

        return CoNLL2005Sentence(
            document_id,
            sentence_id,
            sentence,
            pos_tags,
            dep_parse_head,
            dep_parse_rel,
            predicate_lemmas,
            named_entities,
            srl_frames,
        )


    @staticmethod
    def _process_span_annotations_for_word(
        annotations: List[str],
        span_labels: List[List[str]],
        current_span_labels: List[Optional[str]],
    ) -> None:
        """
        Given a sequence of different label types for a single word and the current
        span label we are inside, compute the BIO tag for each label and append to a list.
        Parameters
        ----------
        annotations: ``List[str]``
            A list of labels to compute BIO tags for.
        span_labels : ``List[List[str]]``
            A list of lists, one for each annotation, to incrementally collect
            the BIO tags for a sequence.
        current_span_labels : ``List[Optional[str]]``
            The currently open span per annotation type, or ``None`` if there is no open span.
        """
        for annotation_index, annotation in enumerate(annotations):
            # strip all bracketing information to
            # get the actual propbank label.
            label = annotation.strip("()*")

            if "(" in annotation:
                # Entering into a span for a particular semantic role label.
                # We append the label and set the current span for this annotation.
                bio_label = "B-" + label
                span_labels[annotation_index].append(bio_label)
                current_span_labels[annotation_index] = label
            elif current_span_labels[annotation_index] is not None:
                # If there's no '(' token, but the current_span_label is not None,
                # then we are inside a span.
                bio_label = "I-" + current_span_labels[annotation_index]
                span_labels[annotation_index].append(bio_label)
            else:
                # We're outside a span.
                span_labels[annotation_index].append("O")
            # Exiting a span, so we reset the current span label for this annotation.
            if ")" in annotation:
                current_span_labels[annotation_index] = None