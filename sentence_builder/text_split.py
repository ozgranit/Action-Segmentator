import os
import word2vec
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from segmentation_metrics import break_seq_p_k, get_casas_data, get_aruba_data
from segmentation_metrics import break_seq_wd, precision_recall
from textsplit.tools import get_penalty, get_segments
from textsplit.algorithm import split_optimal, split_greedy, get_total
import numpy as np

if __name__ == '__main__':
    TP, TN, TotalP, totalN, FP, FN = 0, 0, 0, 0, 0, 0
    TP_lst, TN_lst, TotalP_lst, totalN_lst, FP_lst, FN_lst = [], [], [], [], [], []
    index_to_run_to = 0
    for chunk_counter in range(100):
        current_size = pred[-1] // 100
        index_to_run_to += current_size
        n_gram_pred_array = np.full((current_size, current_size), False, dtype=bool)
        prev = 0
        for i in pred:
            if i > index_to_run_to:
                n_gram_pred_array[prev:index_to_run_to, prev:index_to_run_to] = 1
                break
            n_gram_pred_array[prev:i, prev:i] = 1
            prev = i

        true_array = np.full((current_size, current_size), False, dtype=bool)
        for i in true:
            if i > index_to_run_to:
                true_array[prev:index_to_run_to, prev:index_to_run_to] = 1
                break
            true_array[prev:i, prev:i] = 1
            prev = i
        TotalP_lst.append(n_gram_pred_array.sum())
        totalN_lst.append(np.logical_not(n_gram_pred_array).sum())
        TP_lst.append(np.logical_and(n_gram_pred_array, true_array).sum())
        TN_lst.append(np.logical_not(np.logical_or(n_gram_pred_array, true_array)).sum())
        FN_lst.append(totalN_lst[-1] - TN_lst[-1])
        FP_lst.append(TotalP_lst[-1] - TP_lst[-1])

    fs=[]
    for i in range(len(FN_lst)):
        true_positives = TP_lst[i]
        false_positives = FP_lst[i]
        true_negatives = TN_lst[i]
        false_negatives = FP_lst[i]

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        acc = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        F_score = 2 * ((precision * recall) / (precision + recall))

        fs.append(F_score)

    f_score = np.mean(fs)

    casas_folder_path = Path(os.path.dirname(__file__)) / 'data'
    true_sent_breaks, casas_df = get_aruba_data(casas_folder_path, labels=1500)

    corpus_path = './text.txt'
    with open(corpus_path, "w") as text_file:
        for item in casas_df["Description_ID"]:
            text_file.write(str(item) + " ")

    wrdvec_path = '/groups/pupko/alburquerque/ActionSeg/Action-Segmentator/sentence_builder/wrdvecs.bin'
    # optimal params from grid search
    # wrdvec_path = 'wrdvecs.bin'
    # optimal params from grid search
    # word2vec.word2vec(corpus_path, wrdvec_path, cbow=1, iter_=5, hs=1, threads=8, sample='1e-5', window=7, size=5, binary=1)
    model = Word2Vec( cbow_mean=1,  hs=1, sample=1e-5, window=7, corpus_file=corpus_path, vector_size=5)

    # os.remove(wrdvec_path)
    # os.remove(corpus_path)

    wrdvecs = pd.DataFrame(model.wv.vectors, index=model.wv.key_to_index.keys())
    del model
    print(wrdvecs.shape)

    # optimal params from grid search
    segment_len = 250  # segment target length in sentences
    sentenced_text = [str(i) for i in casas_df["Description_ID"]]

    vecr = CountVectorizer(vocabulary=wrdvecs.index)

    sentence_vectors = vecr.transform(sentenced_text).dot(wrdvecs)

    # # to test with no count CountVectorizer
    # a = []
    # for i in casas_df['Description_ID'].values:
    #     if str(i) not in wrdvecs.index:
    #         print(i)
    #         continue
    #     a.append(pd.DataFrame(wrdvecs.loc[str(i)]).T)
    # df = pd.concat(a)
    # sentence_vectors = df.to_numpy().dot(wrdvecs.T)

    penalty = get_penalty([sentence_vectors], segment_len)
    print('penalty %4.2f' % penalty)

    optimal_segmentation = split_optimal(sentence_vectors, penalty, seg_limit=600)
    segmented_text = get_segments(sentenced_text, optimal_segmentation)

    print('%d sentences, %d segments, avg %4.2f sentences per segment' % (
        len(sentenced_text), len(segmented_text), len(sentenced_text) / len(segmented_text)))

    greedy_segmentation = split_greedy(sentence_vectors, max_splits=len(optimal_segmentation.splits))
    greedy_segmented_text = get_segments(sentenced_text, greedy_segmentation)
    lengths_optimal = [len(segment) for segment in segmented_text for sentence in segment]
    lengths_greedy = [len(segment) for segment in greedy_segmented_text for sentence in segment]
    # df = pd.DataFrame({'greedy': lengths_greedy, 'optimal': lengths_optimal})
    # df.plot.line(figsize=(18, 3), title='Segment lenghts over text')
    # df.plot.hist(bins=30, alpha=0.5, figsize=(10, 3), title='Histogram of segment lengths')

    totals = [get_total(sentence_vectors, seg.splits, penalty)
              for seg in [optimal_segmentation, greedy_segmentation]]
    # print('optimal score %4.2f, greedy score %4.2f' % tuple(totals))
    # print('ratio of scores %5.4f' % (totals[0] / totals[1]))
    #
    # print('greedy segmentation scores:')
    predicted_sentences = sorted(greedy_segmentation.splits) + [len(casas_df)]
    true_sentences = sorted(true_sent_breaks)
    with open("/groups/pupko/alburquerque/ActionSeg/Action-Segmentator/sentence_builder/greedy_dp_res.txt", 'w+') as fp:
        fp.write("predicted_sentences\n\n")
        fp.write(str(predicted_sentences))
        fp.write("\n\n")
        fp.write("true_sentences\n\n")
        fp.write(str(true_sentences))
    # print(f'\tPk score is {break_seq_p_k(predicted_sentences, true_sentences):.2f}\n'
    #       f'\tWD score is {break_seq_wd(predicted_sentences, true_sentences):.2f}')
    # precision, recall, acc, F_score = precision_recall(true_sentences, predicted_sentences, len(casas_df))
    # print(f'\t precision: {precision}, recall: {recall}, acc: {acc}, F_score: {F_score}')
    #
    # print('optimal segmentation scores:')
    predicted_sentences = sorted(optimal_segmentation.splits) + [len(casas_df)]
    with open("/groups/pupko/alburquerque/ActionSeg/Action-Segmentator/sentence_builder/optimal_dp_res.txt", 'w+') as fp:
        fp.write("predicted_sentences\n\n")
        fp.write(str(predicted_sentences))
        fp.write("\n\n")
        fp.write("true_sentences\n\n")
        fp.write(str(true_sentences))
    # true_sentences = sorted(true_sent_breaks)
    # print(f'\tPk score is {break_seq_p_k(predicted_sentences, true_sentences):.2f}\n'
    #       f'\tWD score is {break_seq_wd(predicted_sentences, true_sentences):.2f}')
    # precision, recall, acc, F_score = precision_recall(true_sentences, predicted_sentences, len(casas_df))
    # print(f'\t precision: {precision}, recall: {recall}, acc: {acc}, F_score: {F_score}')

    # plt.show()
