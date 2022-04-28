import os
import numpy as np
import pandas as pd

from nltk.lm import MLE
from pathlib import Path
from nltk.lm.preprocessing import padded_everygram_pipeline
from segmentation_metrics import break_seq_p_k, get_casas_data, get_aruba_data
from segmentation_metrics import precision_recall


class NgramSentenceBuilder:
    def __init__(self, actions: pd.Series, window_size, break_percentile):
        # we expect a column from df
        if not isinstance(actions, pd.Series):
            raise TypeError

        self.actions = actions
        self.break_pts = None
        self.window_size = window_size
        self.break_percentile = break_percentile

    def build_sentences(self):
        window_size = self.window_size
        sentence_break_percentile = self.break_percentile

        actions_str = [str(i) for i in self.actions]
        # Preprocess the tokenized text for 3-grams language modelling
        train_data, padded_sents = padded_everygram_pipeline(window_size, [actions_str])
        model = MLE(window_size)
        model.fit(train_data, padded_sents)
        # print(model.vocab)
        # print(model.counts)

        predicted_prob = []
        for i in range(len(actions_str) - window_size + 1):
            window = actions_str[i: i + window_size]
            predicted_prob.append(model.score(window[-1], window[:-1]))

        sentence_break_threshold = np.percentile(predicted_prob, sentence_break_percentile)

        sent_break = predicted_prob <= sentence_break_threshold
        self.break_pts = [i for i, x in enumerate(sent_break) if x] + [len(self.actions)]

        return self.break_pts


def ngram_results(dataset_getter, data_name, opt_ws, opt_bp):

    folder_path = Path(os.path.dirname(__file__)) / 'data'
    true_sent_breaks, casas_df = dataset_getter(folder_path)

    model = NgramSentenceBuilder(casas_df["Description_ID"], window_size=opt_ws, break_percentile=opt_bp)
    model.build_sentences()

    predicted_break_pts = model.break_pts
    true_sentences = sorted(true_sent_breaks)
    with open("/groups/pupko/alburquerque/ActionSeg/Action-Segmentator/sentence_builder/n_gram_res.txt", 'w+') as fp:
        fp.write("predicted_break_pts\n\n")
        fp.write(str(predicted_break_pts))
        fp.write("\n\n")
        fp.write("true_sentences\n\n")
        fp.write(str(true_sentences))
    p_k = break_seq_p_k(predicted_break_pts, true_sentences)
    f_score = precision_recall(true_sentences, predicted_break_pts, len(casas_df))[3]

    print(f'Ngram check, dataset: {data_name}')
    print(f'p_k: {p_k}, f_score: {f_score}')


if __name__ == '__main__':
    # ngram_results(get_casas_data, 'Kyoto', opt_ws=5, opt_bp=0.05)
    ngram_results(get_aruba_data, 'Aruba', opt_ws=10, opt_bp=0.01)
