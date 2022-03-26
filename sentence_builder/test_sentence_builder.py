import os
import random
import unittest
import itertools
import pandas as pd

from pathlib import Path


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

        # model = NGramModel(casas_df, self.export_path, self.export_path)
        sent_break_index = model.run(verbose=0)

        predicted_sentences = sorted(sent_break_index) + [len(casas_df)]
        true_sentences = sorted(true_sent_breaks)
        print('Pk score is ' + str(break_seq_p_k(predicted_sentences, true_sentences)))
        print('WD score is ' + str(break_seq_wd(predicted_sentences, true_sentences)))


if __name__ == '__main__':
    builder = SentenceBuilder()
    builder.test_casas()
