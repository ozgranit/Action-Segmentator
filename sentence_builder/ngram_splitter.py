import numpy as np
import pandas as pd

from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline


class NgramSentenceBuilder:
    def __init__(self, actions: pd.Series):
        # we expect a column from df
        if not isinstance(actions, pd.Series):
            raise TypeError

        self.actions = actions
        self.break_pts = None

    def build_sentences(self):
        window_size = 5
        sentence_break_percentile = 5

        actions_str = [str(i) for i in self.actions]
        # Preprocess the tokenized text for 3-grams language modelling
        train_data, padded_sents = padded_everygram_pipeline(window_size, [actions_str])
        model = MLE(window_size)
        model.fit(train_data, padded_sents)
        print(model.vocab)
        print(model.counts)

        predicted_prob = []
        for i in range(len(actions_str) - window_size + 1):
            window = actions_str[i: i + window_size]
            predicted_prob.append(model.score(window[-1], window[:-1]))

        sentence_break_threshold = np.percentile(predicted_prob, sentence_break_percentile)

        sent_break = predicted_prob <= sentence_break_threshold
        self.break_pts = [i for i, x in enumerate(sent_break) if x] + [len(self.actions)]

        return self.break_pts

