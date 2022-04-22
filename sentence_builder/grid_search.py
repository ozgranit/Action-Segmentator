import os
import word2vec
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from textsplit.tools import get_penalty
from ngram_splitter import NgramSentenceBuilder
from segmentation_metrics import precision_recall
from textsplit.algorithm import split_optimal, split_greedy
from sklearn.feature_extraction.text import CountVectorizer
from segmentation_metrics import break_seq_p_k, get_casas_data, get_aruba_data


def grid_search(dataset_getter, label_range, data_name):
    data_folder = Path(os.path.dirname(__file__)) / 'data'
    Path(f"./figures_{data_name}").mkdir(exist_ok=True)

    possible_values = {
        'labels': [7] + [i for i in label_range],
        'segment_len': [15, 30, 45, 60, 75, 90, 105, 150],
        'window': [3, 5, 15, 30, 50],
        'cbow': [0, 1],
        'size': [15, 25, 50, 150, 250, 300]
    }

    optimal_values = {  # a result of grid searching
        'cbow': 1,
        'hs': 1,
        'iter_': 5,
        'size': 200,
        'window': 15,
        'segment_len': 30,
        'labels': 7
    }
    names = ['labels', 'segment_len', 'window', 'cbow', 'size']

    naming_transformer = {
        'labels': 'Number of Buckets',
        'cbow': 'CBOW vs Skipgram',
        'window': 'Window size',
        'size': 'Output vector size',
        'segment_len': 'Segment length'
    }

    min_p_k_reached = 1
    max_f_reached = 0
    p_lst = []
    f_lst = []
    optimal_p_lst = []
    optimal_f_lst = []

    for param_to_test in names:
        param_values, param_results_opt, param_results_greedy, param_res_f_opt, param_res_f_greedy = [], [], [], [], []
        for value in possible_values.get(param_to_test):
            current_params = optimal_values.copy()
            current_params[param_to_test] = value
            print(f"Testing {param_to_test} for value {value}")

            true_sent_breaks, casas_df = dataset_getter(data_folder, labels=current_params.get('labels'))
            # params = cbow, hs, iter_, size, window, segment_len, labels
            params = tuple(current_params.values())
            p_k, greedy_pk, f_greedy, f_opt = semantic_split(casas_df, true_sent_breaks, params)

            if p_k < min_p_k_reached:
                optimal_values[param_to_test] = value
                min_p_k_reached = p_k
            max_f_reached = max(f_opt, max_f_reached)

            p_lst.append(p_k)
            f_lst.append(f_opt)
            optimal_p_lst.append(min_p_k_reached)
            optimal_f_lst.append(max_f_reached)

            param_res_f_opt.append(f_opt)
            param_res_f_greedy.append(f_greedy)
            param_values.append(value)
            param_results_opt.append(p_k)
            param_results_greedy.append(greedy_pk)

        # plot results for param
        plt.title(f'Parameter effect on PK - {naming_transformer.get(param_to_test, param_to_test)}')
        plt.xlabel('Parameter value')
        plt.ylabel('Score')
        plt.plot(param_values, param_results_opt, label='optimal pk')
        plt.plot(param_values, param_res_f_opt, label='optimal F score')
        plt.plot(param_values, param_res_f_greedy, label='greedy F score')
        plt.plot(param_values, param_results_greedy, label='greedy pk')
        plt.legend()
        plt.savefig(f'./figures_{data_name}/param_changes_{naming_transformer.get(param_to_test, param_to_test)}.png')
        plt.clf()

    # plot results for entire run
    plt.title(f'Parameter effect on PK')
    plt.xlabel('Experiment')
    plt.ylabel('Score')
    plt.plot(optimal_p_lst, label=f'min pk={optimal_p_lst[-1]:.3f}')
    plt.plot(p_lst, label='pk')
    plt.plot(optimal_f_lst, label=f'max F={optimal_f_lst[-1]:.3f}')
    plt.plot(f_lst, label='F score')
    plt.legend()
    plt.savefig(f'./figures_{data_name}/grid_search_{data_name}.png')
    plt.clf()

    print(f'dataset: {data_name}')
    for name in names:
        print(f'param: {name}, value: {optimal_values[name]}')


def semantic_split(casas_df, true_sent_breaks, params):
    corpus_path = './text.txt'
    with open(corpus_path, "w") as text_file:
        for item in casas_df["Description_ID"]:
            text_file.write(str(item) + " ")

    cbow, hs, iter_, size, window, segment_len, labels = params
    wrdvec_path = 'wrdvecs.bin'
    word2vec.word2vec(corpus_path, wrdvec_path, cbow=cbow, iter_=iter_, hs=hs, threads=8, sample='1e-5', window=window,
                      size=size, binary=1)

    model = word2vec.load(wrdvec_path)
    wrdvecs = pd.DataFrame(model.vectors, index=model.vocab)
    del model
    os.remove(corpus_path)
    os.remove(wrdvec_path)

    sentenced_text = [str(i) for i in casas_df["Description_ID"]]

    vecr = CountVectorizer(vocabulary=wrdvecs.index)
    sentence_vectors = vecr.transform(sentenced_text).dot(wrdvecs)

    penalty = get_penalty([sentence_vectors], segment_len)

    optimal_segmentation = split_optimal(sentence_vectors, penalty, seg_limit=250)
    greedy_segmentation = split_greedy(sentence_vectors, max_splits=len(optimal_segmentation.splits))

    optimal_predicted_sentences = sorted(optimal_segmentation.splits) + [len(casas_df)]
    greedy_predicted_sentences = sorted(greedy_segmentation.splits) + [len(casas_df)]
    true_sentences = sorted(true_sent_breaks)

    optimal_pk = break_seq_p_k(optimal_predicted_sentences, true_sentences)
    greedy_pk = break_seq_p_k(greedy_predicted_sentences, true_sentences)
    f_greedy = precision_recall(true_sentences, greedy_predicted_sentences, len(casas_df))[3]
    f_opt = precision_recall(true_sentences, optimal_predicted_sentences, len(casas_df))[3]

    print(f'\tPk score is {optimal_pk:.2f}\n')

    return optimal_pk, greedy_pk, f_greedy, f_opt


def ngram_grid_search(dataset_getter, data_name):

    folder_path = Path(os.path.dirname(__file__)) / 'data'
    true_sent_breaks, casas_df = dataset_getter(folder_path)
    window_sizes = [3, 5, 7, 10, 15]
    break_percentiles = [0.01, 0.05, 0.5, 5, 10, 50]

    min_p_k_reached = 1
    p_lst = []
    optimal_p_lst = []

    opt_bp = 0.5
    opt_ws = 5

    for bp in break_percentiles:
        model = NgramSentenceBuilder(casas_df["Description_ID"], window_size=opt_ws, break_percentile=bp)
        model.build_sentences()

        predicted_break_pts = model.break_pts
        true_sentences = sorted(true_sent_breaks)
        p_k = break_seq_p_k(predicted_break_pts, true_sentences)

        if p_k < min_p_k_reached:
            opt_bp = bp
            min_p_k_reached = p_k
        p_lst.append(p_k)
        optimal_p_lst.append(min_p_k_reached)

    for ws in window_sizes:
        model = NgramSentenceBuilder(casas_df["Description_ID"], window_size=ws, break_percentile=opt_bp)
        model.build_sentences()

        predicted_break_pts = model.break_pts
        true_sentences = sorted(true_sent_breaks)
        p_k = break_seq_p_k(predicted_break_pts, true_sentences)

        if p_k < min_p_k_reached:
            opt_ws = ws
            min_p_k_reached = p_k
        p_lst.append(p_k)
        optimal_p_lst.append(min_p_k_reached)

    print(f'Ngram check, dataset: {data_name}')
    print(f'window_size: {opt_ws}, break_percentile: {opt_bp}')

    # plot results for entire run
    plt.title(f'Ngram Parameters')
    plt.xlabel('Experiment')
    plt.ylabel('Pk')
    plt.plot(optimal_p_lst, label=f'min pk={optimal_p_lst[-1]:.3f}')
    plt.plot(p_lst, label='pk')
    plt.legend()
    plt.savefig(f'./figures_{data_name}/ngram_params.png')
    plt.clf()


if __name__ == '__main__':
    grid_search(get_casas_data, [3, 5, 10, 15, 50, 100, 200, 400, 570, 1050], 'Kyoto')
    grid_search(get_aruba_data, [3, 5, 10, 15, 50, 100, 570, 10000, 20000, 30000], 'Aruba')

    ngram_grid_search(get_casas_data, 'Kyoto')
    ngram_grid_search(get_aruba_data, 'Aruba')
