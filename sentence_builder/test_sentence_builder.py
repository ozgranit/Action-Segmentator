import os
import random
import unittest
import itertools
import pandas as pd
from textsplit.tools import get_penalty, get_segments
from textsplit.algorithm import split_optimal, split_greedy, get_total
from pathlib import Path
import word2vec
from sentence_builder.segmentation_metrics import get_casas_data, break_seq_p_k, break_seq_wd
from textsplit.tools import SimpleSentenceTokenizer
from sklearn.feature_extraction.text import CountVectorizer


class SentenceBuilder:
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


        # original word2vec
        corpus_path = './text.txt'
        with open(corpus_path, "w") as text_file:
            for item in casas_df["Description_ID"]:
                text_file.write(str(item) + " ")

        wrdvec_path = 'wrdvecs.bin'
        if not os.path.exists(wrdvec_path):
            word2vec.word2vec(corpus_path, wrdvec_path, cbow=1, iter_=5, hs=1, threads=8, sample='1e-5', window=15,
                              size=200, binary=1)

        model = word2vec.load(wrdvec_path)
        wrdvecs = pd.DataFrame(model.vectors, index=model.vocab)

        vecr = CountVectorizer(vocabulary=wrdvecs.index)
        sentenced_text = [str(i) for i in casas_df["Description_ID"]]
        sentence_vectors = vecr.transform(sentenced_text).dot(wrdvecs)
        #
        # a = []
        # for i in casas_df['Description_ID'].values:
        #     if str(i) not in wrdvecs.index:
        #         print(i)
        #         continue
        #     a.append(pd.DataFrame(wrdvecs.loc[str(i)]).T)
        # df = pd.concat(a)

        # word_context_vectors = df.to_numpy().dot(wrdvecs.T)
        target_seg_len_in_sent = 30
        penalty = get_penalty([sentence_vectors], target_seg_len_in_sent)
        optimal_segmentation = split_optimal(sentence_vectors, penalty, seg_limit=250)

        predicted_sentences = sorted(optimal_segmentation.splits) + [len(casas_df)]
        true_sentences = sorted(true_sent_breaks)
        print('Pk score is ' + str(break_seq_p_k(predicted_sentences, true_sentences)))
        print('WD score is ' + str(break_seq_wd(predicted_sentences, true_sentences)))


if __name__ == '__main__':
    builder = SentenceBuilder()
    builder.test_casas()
