import os
import numpy as np
import pandas as pd
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


def get_aruba_data(folder_path):
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
    seq["sensor_state"][num_mask] = pd.cut(num_sensor_state, bins=7, labels=np.arange(7), right=False)

    seq['Description_parsed'] = seq["sensor_name"] + seq["sensor_state"].astype(str)
    le = preprocessing.LabelEncoder()
    labels_seq = le.fit_transform(seq.Description_parsed.values)
    seq["Description_ID"] = labels_seq

    return sent_breaks, seq


def get_casas_data(folder_path):
    """returns seqs of casas dataset"""

    df_lst = []
    for file in os.listdir(folder_path / 'adlnormal'):
        filename = os.fsdecode(file)
        if filename.startswith("p"):
            df = get_df(folder_path / filename)
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
    seq["sensor_state"][num_mask] = pd.cut(num_sensor_state, bins=7, labels=np.arange(7), right=False)

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


if __name__ == '__main__':
    from pathlib import Path
    get_aruba_data(Path(os.path.dirname(__file__)) / 'data')
