import os
import sys

from gensim.models import word2vec
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from segmentation_metrics import break_seq_p_k, get_casas_data, get_aruba_data
from segmentation_metrics import break_seq_wd, precision_recall
from textsplit.tools import get_penalty, get_segments
from textsplit.algorithm import split_optimal, split_greedy, get_total
import itertools
import flatten_json
import joblib
import time
if __name__ == '__main__':

    casas_folder_path = Path(os.path.dirname(__file__)) / 'data'
    # true_sent_breaks, casas_df = get_casas_data(casas_folder_path)
    true_sent_breaks, casas_df = get_aruba_data(casas_folder_path, labels=1500)
    # which is ok cause is where true sent breaks anyway
    corpus_path = './text_2.txt'
    with open(corpus_path, "w") as text_file:
        for item in casas_df["Description_ID"]:
            text_file.write(str(item) + " ")


    possible_values = {
    'segment_len': [150,200,250,300,350,400],
    'cbow': [],
    'negative':[],
    'iter_': [],
    'hs': [],
    'sample' :  [],
    'window' :[5,10,15,25,40,60,80],
    'size': [5, 10, 20, 30, 40, 50, 75, 100, 150, 200,250,300,400],
    'binary':[]
    }
    results = {}

    optimal_values = {  # a result of grid searching
        'binary': 1,
        'cbow': 1,
        'hs': 1,
        'iter_': 5,
        'negative': 3,
        'sample': 1e-5,
        'segment_len': 15,
        'size': 150,
        'window': 15
    }

    naming_transformer = {
        'negative': 'Negative sampling rate',
        'sample': 'Occurrence threshold',
        'window': 'Window size',
        'size': 'Output vector size',
        'segment_len': 'Segment length'
    }

    allNames = sorted(possible_values)
    temp_scores = []
    best_pk = sys.maxsize
    Path("figures_Aruba_new").mkdir(exist_ok=True)
    k=0
    for param_to_test in allNames:
        param_values, param_results_opt, param_results_greedy, param_res_f_opt, param_res_f_greedy = [], [], [], [], []
        if param_to_test in ('binary','hs','iter_','cbow'):
            continue
        for value in possible_values.get(param_to_test):
            current_params = optimal_values.copy()
            current_params[param_to_test] = value
            binary, cbow, hs, iter_, negative, sample, segment_len, size, window = [current_params.get(key) for key in allNames]
            # train with this param

            model = word2vec.Word2Vec(corpus_file=corpus_path, cbow_mean=cbow, hs=hs, sample=1e-5, window=window,
                                      vector_size=size)

            wrdvecs = pd.DataFrame(model.wv.vectors, index=model.wv.key_to_index.keys())
            del model
            sentenced_text = [str(i) for i in casas_df["Description_ID"]]
            vecr = CountVectorizer(vocabulary=wrdvecs.index)
            sentence_vectors = vecr.transform(sentenced_text).dot(wrdvecs)
            penalty = get_penalty([sentence_vectors], segment_len)
            optimal_segmentation = split_optimal(sentence_vectors, penalty, seg_limit=600)
            segmented_text = get_segments(sentenced_text, optimal_segmentation)
            greedy_segmentation = split_greedy(sentence_vectors, max_splits=len(optimal_segmentation.splits))
            true_sentences = sorted(true_sent_breaks)
            greedy_predicted_sentences = sorted(greedy_segmentation.splits) + [len(casas_df)]
            optimal_predicted_sentences = sorted(optimal_segmentation.splits) + [len(casas_df)]
            optimal_pk = break_seq_p_k(optimal_predicted_sentences, true_sentences)
            greedy_pk = break_seq_p_k(greedy_predicted_sentences, true_sentences)
            f_greedy = precision_recall(true_sentences, greedy_predicted_sentences, len(casas_df))[3]
            f_opt = precision_recall(true_sentences, optimal_predicted_sentences, len(casas_df))[3]

            param_res_f_opt.append(f_opt)
            param_res_f_greedy.append(f_greedy)
            param_values.append(value)
            param_results_opt.append(optimal_pk)
            param_results_greedy.append(greedy_pk)
            print(k)
            k+=1

        # plot results for param

        plt.title(f'Parameter effect on PK - {naming_transformer.get(param_to_test, param_to_test)}')
        plt.xlabel('Parameter value')
        plt.ylabel('PK Score')
        plt.plot(param_values, param_results_opt, label='optimal pk')
        plt.plot(param_values, param_res_f_opt, label='optimal F score')
        plt.plot(param_values, param_res_f_greedy, label='greedy F score')
        plt.plot(param_values, param_results_greedy, label='greedy pk')
        plt.legend()
        plt.savefig(f'./figures/param_changes_{naming_transformer.get(param_to_test, param_to_test)}.png')
        plt.clf()
