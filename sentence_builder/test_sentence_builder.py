import os
import random
import unittest
import itertools
import pandas as pd
from textsplit.tools import get_penalty, get_segments
from textsplit.algorithm import split_optimal, split_greedy, get_total
from pathlib import Path

from sentence_builder.segmentation_metrics import get_casas_data, break_seq_p_k, break_seq_wd


class SentenceBuilder(unittest.TestCase):
    """Sentence segmentation on generated sentences"""

    def test_casas(self):
        """
        To evaluate our performance we use the Pk and the WindowDiff (WD) metric,
        both described in [1], on the CASAS smart home project dataset.
        This dataset contains natural human routines.

        Related Works:
        [1] https://arxiv.org/pdf/1503.05543.pdf - Text Segmentation based on Semantic Word Embeddings
        [2] http://casas.wsu.edu/datasets/
        """
        casas_folder_path = Path(os.path.dirname(__file__)) / 'data' / 'adlnormal'
        true_sent_breaks, casas_df = get_casas_data(casas_folder_path)

        sentence_vectors = casas_df['Description_ID']
        sentence_vectors = sentence_vectors.to_numpy().reshape(-1, 1)
        # todo: why is the penalty always zero
        target_seg_len_in_sent = 50
        # penalty = get_penalty(sentence_vectors, target_seg_len_in_sent)
        penalty = 0.2
        optimal_segmentation = split_optimal(sentence_vectors, penalty, seg_limit=target_seg_len_in_sent)

        # model = NGramModel(casas_df, self.export_path, self.export_path)
        # sent_break_index = model.run(verbose=0)

        predicted_sentences = sorted(optimal_segmentation.splits) + [len(casas_df)]
        true_sentences = sorted(true_sent_breaks)
        print('Pk score is ' + str(break_seq_p_k(predicted_sentences, true_sentences)))
        print('WD score is ' + str(break_seq_wd(predicted_sentences, true_sentences)))


if __name__ == '__main__':
    builder = SentenceBuilder()
    builder.test_casas()
