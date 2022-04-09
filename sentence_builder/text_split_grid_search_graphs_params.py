import os
import sys

import word2vec
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from segmentation_metrics import break_seq_p_k, get_casas_data
from segmentation_metrics import break_seq_wd, precision_recall
from textsplit.tools import get_penalty, get_segments
from textsplit.algorithm import split_optimal, split_greedy, get_total
import itertools
import flatten_json
import joblib

if __name__ == '__main__':

    casas_folder_path = Path(os.path.dirname(__file__)) / 'data' / 'adlnormal'
    true_sent_breaks, casas_df = get_casas_data(casas_folder_path)
    corpus_path = './text_2.txt'
    with open(corpus_path, "w") as text_file:
        for item in casas_df["Description_ID"]:
            text_file.write(str(item) + " ")


    possible_values = {
    'segment_len': [5, 10, 15, 20, 25, 27, 30, 35, 40, 45, 50, 55, 60, 65],
    'cbow': [0, 1],
    'negative':[0, 3, 5, 8, 10, 13, 15, 20],
    'iter_': [5, 10],
    'hs': [0, 1],
    'sample' :  ['0', '1e-3', '1e-4', '1e-5', '1e-6'],
    'window' :[5, 10, 15, 25, 30, 35, 50, 100],
    'size': [5, 10, 20, 30, 40, 50, 75, 100, 150, 200],
    'binary':[1]
    }
    results = {}

    optimal_values = {
    'segment_len': 30,
    'cbow': 1,
    'negative': 0,
    'iter_': 5,
    'hs': 1,
    'sample' : '1e-5',
    'window' : 15,
    'size': 200,
    'binary':1
    }

    allNames = sorted(possible_values)
    temp_scores = []
    best_pk = sys.maxsize
    Path("./figures").mkdir(exist_ok=True)

    for param_to_test in allNames:
        param_values, param_results_opt, param_results_greedy, param_res_precision_opt, param_res_precision_greedy = [], [], [], [], []
        if param_to_test in ('binary','hs','iter_','cbow'):
            continue
        for value in possible_values.get(param_to_test):
            current_params = optimal_values.copy()
            current_params[param_to_test] = value
            binary, cbow, hs, iter_, negative, sample, segment_len, size, window = [current_params.get(key) for key in allNames]
            # train with this param
            wrdvec_path = 'wrdvecs_graphs.bin'
            word2vec.word2vec(corpus_path, wrdvec_path, cbow=cbow, iter_=iter_, hs=hs, threads=8, sample=sample,
                              window=window, size=size, binary=binary, negative=negative)
            model = word2vec.load(wrdvec_path)
            wrdvecs = pd.DataFrame(model.vectors, index=model.vocab)
            del model
            sentenced_text = [str(i) for i in casas_df["Description_ID"]]
            vecr = CountVectorizer(vocabulary=wrdvecs.index)
            sentence_vectors = vecr.transform(sentenced_text).dot(wrdvecs)
            penalty = get_penalty([sentence_vectors], segment_len)
            optimal_segmentation = split_optimal(sentence_vectors, penalty, seg_limit=250)
            segmented_text = get_segments(sentenced_text, optimal_segmentation)
            greedy_segmentation = split_greedy(sentence_vectors, max_splits=len(optimal_segmentation.splits))
            true_sentences = sorted(true_sent_breaks)
            greedy_predicted_sentences = sorted(greedy_segmentation.splits) + [len(casas_df)]
            optimal_predicted_sentences = sorted(optimal_segmentation.splits) + [len(casas_df)]
            optimal_pk = break_seq_p_k(optimal_predicted_sentences, true_sentences)
            greedy_pk = break_seq_p_k(greedy_predicted_sentences, true_sentences)
            precision_greedy = precision_recall(true_sentences, greedy_predicted_sentences, len(casas_df))[0]
            precision_opt = precision_recall(true_sentences, optimal_predicted_sentences, len(casas_df))[0]

            param_res_precision_opt.append(precision_opt)
            param_res_precision_greedy.append(precision_greedy)
            param_values.append(value)
            param_results_opt.append(optimal_pk)
            param_results_greedy.append(greedy_pk)

        # plot results for param

        plt.title(f'Parameter effect on PK - {param_to_test}')
        plt.xlabel('Parameter value')
        plt.ylabel('PK Score')
        plt.plot(param_values, param_results_opt, label='optimal pk')
        plt.plot(param_values, param_res_precision_opt, label='optimal precision')
        plt.plot(param_values, param_res_precision_greedy, label='greedy precision')
        plt.plot(param_values, param_results_greedy, label='greedy pk')
        plt.legend()
        plt.savefig(f'./figures/param_changes_{param_to_test}.png')
        plt.clf()
