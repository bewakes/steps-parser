#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

from torch.utils.data import Dataset

from math import inf
import os
import itertools
from pathlib import Path

from data_handling.dependency_matrix import DependencyMatrix
from data_handling.tag_sequence import TagSequence
from data_handling.vocab import BasicVocab
from data_handling.annotated_sentence import AnnotatedSentence
from util.color_logger import Logger


def get_lang(fname):
    """Get language id from filename

    Args:
        fname: of the form <lang>.conllu
    """
    if not fname.endswith('.conllu'):
        return
    return fname.split('.')[0]



class CustomCoNLLDataset(Dataset):
    """An object of this class represents a (map-style) dataset of annotated sentences in a CoNLL-like format.
    The individual objects contained within the dataset are of type AnnotatedSentence.
    """
    def __init__(self, langs=[], total_samples_size=None):
        """
        """
        self.langs = langs
        self.total_samples_size = total_samples_size
        self.lang_sentences = {lang: [] for lang in self.langs}
        self.sentences = list()

    def __len__(self):
        if self.langs:
            size = sum([len(lang_sents) for lang_sents in self.lang_sentences.values()])
        else:
            size = len(self.sentences)
        if self.total_samples_size:
            return min(size, self.total_samples_size)

        return len(self.sentences)

    def __getitem__(self, item):
        if not self.sentences and self.langs:
            Logger.warn('No sents present and self.langs. Concatanating sentences of all langs')
            self.sentences = list(itertools.chain(*self.lang_sentences.values()))
        return self.sentences[item]

    def append_sentence(self, sent, lang=None):
        """Append one sentence to the dataset.

        Args:
            sent: AnnotatedSentence object to add to the dataset.
        """
        if lang and self.langs:
            self.lang_sentences[lang].append(sent)
        else:
            self.sentences.append(sent)

    @staticmethod
    def from_corpus_dir(corpus_dirname, annotation_layers, max_sent_len=inf,
            keep_traces=False,multiple_langs=None, samples_per_lang=None,langs=None,total_samples_size=None):
        """Similar to from_corpus_filename except that it reads multiple conllu
        files(multi-lang) from the dir.
        NOTE: This function does not take in subset_size parameter.
        """
        dirlangs = [get_lang(x) for x in os.listdir(corpus_dirname)]
        dirlangs = [x for x in dirlangs if x and (langs is None or x in langs)]

        dataset = CustomCoNLLDataset(langs=dirlangs, total_samples_size=total_samples_size)
        for lang in dirlangs:
            fname = lang + '.conllu'

            fpath = Path(corpus_dirname) / fname
            iterator = _iter_conll_sentences(fpath)
            for i, raw_conll_sent in enumerate(iterator):
                if samples_per_lang and i >= samples_per_lang:
                    break
                processed_sent = AnnotatedSentence.from_conll(raw_conll_sent, annotation_layers, keep_traces=keep_traces)
                if len(processed_sent) <= max_sent_len:
                    dataset.append_sentence(processed_sent, lang)

        return dataset

    @staticmethod
    def from_corpus_file(corpus_filename, annotation_layers, max_sent_len=inf, keep_traces=False, subset_size=None):
        """Read in a dataset from a corpus file in CoNLL format.

        Args:
            corpus_filename: Path to the corpus file to read from.
            annotation_layers: Dictionary mapping annotation IDs to annotation type and CoNLL column to read data from.
            max_sent_len: The maximum length of any given sentence. Sentences with a greater length are ignored.
            keep_traces: Whether to keep empty nodes as tokens (used in enhanced UD; default: False).
            subset_size: If we want only N of the total sentences in corpus file. If None, all will be retrieved.
        Returns:
            A CustomCoNLLDataset object containing the sentences in the input corpus file, with the specified annotation
            layers.
        """
        dataset = CustomCoNLLDataset()

        iterator = _iter_conll_sentences(corpus_filename)
        for i, raw_conll_sent in enumerate(iterator):
            processed_sent = AnnotatedSentence.from_conll(raw_conll_sent, annotation_layers, keep_traces=keep_traces)
            if len(processed_sent) <= max_sent_len:
                dataset.append_sentence(processed_sent)

            if subset_size is not None and i >= subset_size-1:
                break

        return dataset

    @staticmethod
    def extract_label_vocab(*conllu_datasets, annotation_id):
        """Extract a vocabulary of labels from one or more CONLL-U datasets.

        Args:
            *conllu_datasets: One or more CustomCoNLLDataset objects to extract the label.
            annotation_id: Identifier of the annotation layer to extract labels for.
        """
        vocab = BasicVocab()

        for dataset in conllu_datasets:
            for sentence in dataset:
                if isinstance(sentence[annotation_id], DependencyMatrix):
                    for label in [lbl for head_row in sentence[annotation_id].data for lbl in head_row]:
                        vocab.add(label)
                elif isinstance(sentence[annotation_id], TagSequence):
                    for label in sentence[annotation_id].data:
                        vocab.add(label)
                else:
                    raise Exception("Unknown annotation type")

        assert vocab.is_consistent()
        return vocab


def _iter_conll_sentences(conll_file):
    """Helper function to iterate over the CoNLL sentence data in the given file.
        Args:
            conll_file: The custom CoNLL file to parse.
        Yields:
            An iterator over the raw CoNLL lines for each sentence.
    """
    # CoNLL parsing code adapted from https://github.com/pyconll/pyconll/blob/master/pyconll/_parser.py
    opened_file = False
    if isinstance(conll_file, (str, Path)):
        conll_file = open(conll_file, "r")
        opened_file = True

    sent_lines = []
    for line in conll_file:
        line = line.strip()

        if line:
            sent_lines.append(line)
        else:
            if sent_lines:
                yield sent_lines
                sent_lines = []

    if sent_lines:
        yield sent_lines

    if opened_file:
        conll_file.close()
