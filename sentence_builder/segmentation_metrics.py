import os
import word2vec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from random import shuffle
from sklearn import preprocessing


def entire_seq_p_k(predicted_sentences, true_sentences):
    """
    Args:
        true_sentences: a sequence of the form [0,0,..,1,1...2..]
        predicted_sentences: a sequence of the same form, with predicted sentences

    Returns:
        Pk metric as in https://arxiv.org/pdf/1503.05543.pdf - Text Segmentation based on Semantic Word Embeddings
    """

    def delta(i, j, sentences):
        if sentences[i] == sentences[j]:
            return 1
        return 0

    num_of_sentences = true_sentences[-1]
    n = len(true_sentences)
    k = int(0.5 * (n / num_of_sentences)) - 1

    assert n > k > 0
    p_k_score = 0
    for idx in range(n - k):
        if delta(idx, idx + k, true_sentences) != delta(idx, idx + k, predicted_sentences):
            p_k_score += 1

    p_k_score /= (n - k)
    return p_k_score


def break_seq_p_k(hyp, ref, k=None):
    """
    Args:
        hyp, ref: a sequence of the form [3, 5, 8, 10]
        for breaking “aaabbcccdd” into (“aaa”, “bb”, “ccc”, “dd”)
    Returns:
        The Pk metric from Beeferman
    """
    # if k is undefined, use half the mean segment size
    k = k or int(round(0.5 * ref[-1] / len(ref))) - 1
    assert k > 0

    length = ref[-1]
    probeinds = np.arange(length - k)
    dref = np.digitize(probeinds, ref) == np.digitize(probeinds + k, ref)
    dhyp = np.digitize(probeinds, hyp) == np.digitize(probeinds + k, hyp)

    return (dref ^ dhyp).mean()


def break_seq_wd(hyp, ref, k=None):
    """ The window diff metric of Pevzner """
    k = k or int(round(0.5*ref[-1]/len(ref)))-1
    assert k > 0

    length = ref[-1]
    hyp = np.asarray(hyp)
    ref = np.asarray(ref)

    score = 0.0
    tot = 0.0
    for i in range(length - k):
        bref = ((ref > i) & (ref <= i+k)).sum()
        bhyp = ((hyp > i) & (hyp <= i+k)).sum()
        score += 1.0*(np.abs(bref-bhyp) > 0)
        tot += 1.0
    return score/tot


def get_aruba_data(folder_path, labels=12910):
    """returns seqs of casas-aruba dataset"""

    filename = folder_path / 'aruba' / 'aruba_data'
    headers = ["date", "time", "sensor_name", "sensor_state", "activity_name", "activity_state"]
    data = pd.read_csv(filename, index_col=False, names=headers, delim_whitespace=True)
    data['timestamp'] = data["date"] + " " + data["time"]
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    seq = data

    activity_breaks = ~seq['activity_state'].isna()
    sent_breaks = seq.index[activity_breaks].tolist()

    # Label encoder - Description
    num_mask = pd.to_numeric(seq.sensor_state, errors='coerce').notnull()
    num_sensor_state = pd.to_numeric(seq["sensor_state"][num_mask])
    seq["sensor_state"][num_mask] = pd.cut(num_sensor_state, bins=labels, labels=np.arange(labels), right=False)

    seq['Description_parsed'] = seq["sensor_name"] + seq["sensor_state"].astype(str)
    le = preprocessing.LabelEncoder()
    labels_seq = le.fit_transform(seq.Description_parsed.values)
    seq["Description_ID"] = labels_seq

    return sent_breaks, seq


def get_casas_data(folder_path, labels=570):
    """returns seqs of casas dataset"""

    df_lst = []
    for file in os.listdir(folder_path / 'adlnormal'):
        filename = os.fsdecode(file)
        if filename.startswith("p"):
            df = get_df(folder_path / 'adlnormal' / filename)
            df_lst.append(df)
    shuffle(df_lst)

    sent_breaks = []
    last_idx = 0
    for df in df_lst:
        last_idx += len(df.index)
        sent_breaks.append(last_idx)

    seq = pd.concat(df_lst)

    # Label encoder - Description
    num_mask = pd.to_numeric(seq.sensor_state, errors='coerce').notnull()
    num_sensor_state = pd.to_numeric(seq["sensor_state"][num_mask])
    seq["sensor_state"][num_mask] = pd.cut(num_sensor_state, bins=labels, labels=np.arange(labels), right=False)

    seq['Description_parsed'] = seq["sensor_name"] + seq["sensor_state"].astype(str)
    le = preprocessing.LabelEncoder()
    labels_seq = le.fit_transform(seq.Description_parsed.values)
    seq["Description_ID"] = labels_seq

    return sent_breaks, seq


def get_df(filename):

    headers = ["date", "time", "sensor_name", "sensor_state", "activity_name"]
    data = pd.read_csv(filename, header=None, names=headers, delim_whitespace=True)
    data['timestamp'] = data["date"] + " " + data["time"]
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    return data


def precision_recall(true_segs, predicted_segs, text_len):
    true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0

    # build true clustering allocation
    true_seg_allocation = {}
    prev_seg, cluster_index = 0, 0
    for seg_point in true_segs:
        for point in range(prev_seg, seg_point):
            true_seg_allocation[point] = cluster_index
        cluster_index += 1
        prev_seg = seg_point

    # build pred clustering allocation
    pred_seg_allocation = {}
    prev_seg, cluster_index = 0, 0
    for seg_point in predicted_segs:
        for point in range(prev_seg, seg_point):
            pred_seg_allocation[point] = cluster_index
        cluster_index += 1
        prev_seg = seg_point

    for i in range(text_len):
        for j in range(i+1, text_len):
            are_the_same_cluster_true = true_seg_allocation.get(i) == true_seg_allocation.get(j)
            are_the_same_cluster_pred = pred_seg_allocation.get(i) == pred_seg_allocation.get(j)

            if are_the_same_cluster_true is True:
                if are_the_same_cluster_pred is True:
                    true_positives += 1
                else:
                    false_negatives += 1

            if are_the_same_cluster_true is False:
                if are_the_same_cluster_pred is False:
                    true_negatives += 1
                else:
                    false_positives += 1

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    acc = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    F_score = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, acc, F_score


def find_max_vocab(dataset_getter, range_func, data_name):
    folder_path = Path(os.path.dirname(__file__)) / 'data'
    max_vocab = 0
    best_bucket_num = 7
    vocab_list = []
    bucket_options = [i for i in range_func]

    for i in bucket_options:
        true_sent_breaks, casas_df = dataset_getter(folder_path, labels=i)
        corpus_path = './text.txt'
        with open(corpus_path, "w") as text_file:
            for item in casas_df["Description_ID"]:
                text_file.write(str(item) + " ")

        wrdvec_path = 'wrdvecs.bin'
        # optimal params from grid search
        word2vec.word2vec(corpus_path, wrdvec_path, cbow=1, iter_=5, hs=1, threads=8, sample='1e-5', window=15, size=150,
                          binary=1)
        model = word2vec.load(wrdvec_path)

        os.remove(wrdvec_path)
        os.remove(corpus_path)

        wrdvecs = pd.DataFrame(model.vectors, index=model.vocab)
        del model
        vocab, vec_size = wrdvecs.shape
        vocab_list.append(vocab)
        if vocab > max_vocab:
            max_vocab = vocab
            best_bucket_num = i

    print(f'max_vocab = {max_vocab}, best_bucket_num = {best_bucket_num}')
    plt.title(f'# Buckets VS Vocab size')
    plt.xlabel('# Buckets')
    plt.ylabel('Vocab size')
    plt.plot(bucket_options, vocab_list, label='Kyoto data')
    plt.legend()
    plt.savefig(f'./figures/max_vocab_{data_name}.png')
    plt.clf()


if __name__ == '__main__':
    find_max_vocab(get_aruba_data, range(10, 52200, 100), 'Aruba')
    find_max_vocab(get_casas_data, range(10, 1200, 20), 'Kyoto')
