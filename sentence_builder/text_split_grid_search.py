import os
import word2vec
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from segmentation_metrics import break_seq_p_k, get_casas_data
from segmentation_metrics import break_seq_wd
from textsplit.tools import get_penalty, get_segments
from textsplit.algorithm import split_optimal, split_greedy, get_total

import itertools
import flatten_json
import joblib

if __name__ == '__main__':

    casas_folder_path = Path(os.path.dirname(__file__)) / 'data' / 'adlnormal'
    true_sent_breaks, casas_df = get_casas_data(casas_folder_path)
    corpus_path = './text.txt'
    with open(corpus_path, "w") as text_file:
        for item in casas_df["Description_ID"]:
            text_file.write(str(item) + " ")


    possible_values = {
    'segment_len': [30,10,20,40,50,60], # 35,25,65,55,45
    'cbow': [1,0],
    'negative': [5,13,20,3,0], # 8,10,15
    'iter_': [5,10],
    'hs': [1,0],
    'sample' : ['1e-5','0','1e-3'], # ,'1e-4','1e-6'
    'window' :[15,30,5,10], # 25,50,35, 100
    'size': [200,10,50,75,100,150],
    'binary':[1]
    }
    results = {}

    allNames = sorted(possible_values)
    combinations = list(itertools.product(*(possible_values[Name] for Name in allNames)))
    exp_number = 1
    temp_scores = []
    for comb in combinations:
        binary, cbow, hs, iter_, negative, sample, segment_len, size, window = comb
        param_string = f'binary:{binary},cbow:{cbow},hs:{hs},iter_:{iter_},negative:{negative},sample:{sample},segment_len:{segment_len},size:{size},window:{window}'

        wrdvec_path = 'wrdvecs.bin'
        # if not os.path.exists(wrdvec_path):
        word2vec.word2vec(corpus_path, wrdvec_path, cbow=cbow, iter_=iter_, hs=hs, threads=8, sample=sample,
                          window=window, size=size, binary=binary, negative=negative)

        model = word2vec.load(wrdvec_path)
        wrdvecs = pd.DataFrame(model.vectors, index=model.vocab)
        del model
        # print(wrdvecs.shape)

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
        current_score = {
            'greedy_pk': break_seq_p_k(greedy_predicted_sentences, true_sentences),
            'greedy_wd': break_seq_wd(greedy_predicted_sentences, true_sentences),
            'optimal_pk': optimal_pk,
            'optimal_wd': break_seq_wd(optimal_predicted_sentences, true_sentences)
        }
        results[param_string] = current_score
        exp_number += 1
        temp_scores.append(optimal_pk)
        if exp_number % 50 == 0:
            joblib.dump(results, 'results.pkl')
            print(sorted(temp_scores)[0])
            temp_scores = []

    joblib.dump(results, 'results.pkl')
    flat_res = flatten_json.flatten(results)
    sorted_keys = sorted(flat_res, key=flat_res.get)
    print(f'the best key is {sorted_keys[0]} \n with a score of {flat_res.get(sorted_keys[0])}')