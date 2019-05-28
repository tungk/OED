import os
import sys

import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from utils import *


def sliding_dot_product(q, t):
    n = t.size
    m = q.size

    # Append t with n zeros
    ta = np.append(t, np.zeros(n))

    # Reverse Q
    qr = np.flip(q, 0)

    # Append qra
    qra = np.append(qr, np.zeros(2 * n - m))

    # Compute FFTs
    qraf = np.fft.fft(qra)
    taf = np.fft.fft(ta)

    # Compute the inverse FFT to the element-wise multiplication of qraf and taf
    qt = np.fft.ifft(np.multiply(qraf, taf))
    return qt[m:n]


def sliding_dot_product_stomp(q, t):
    n = t.size
    m = q.size

    # Append t with n zeros
    ta = np.append(t, np.zeros(n))

    # Reverse Q
    qr = np.flip(q, 0)

    # Append qra
    qra = np.append(qr, np.zeros(2 * n - m))

    # Compute FFTs
    qraf = np.fft.fft(qra)
    taf = np.fft.fft(ta)

    # Compute the inverse FFT to the element-wise multiplication of qraf and taf
    qt = np.fft.ifft(np.multiply(qraf, taf))
    return qt[m - 1:n]


def calculate_distance_profile(q, t, qt, a, sum_q, sum_q2, mean_t, sigma_t):
    n = t.size
    m = q.size

    b = np.zeros(n - m)
    dist = np.zeros(n - m)
    for i in range(0, n - m):
        b[i] = -2 * (qt[i].real - sum_q * mean_t[i]) / sigma_t[i]
        dist[i] = a[i] + b[i] + sum_q2
    return np.sqrt(np.abs(dist))


# The code below takes O(m) for each subsequence
# you should replace it for MASS
def compute_mean_std_for_query(Q):
    # Compute Q stats -- O(n)
    sumQ = np.sum(Q)
    sumQ2 = np.sum(np.power(Q, 2))
    return sumQ, sumQ2


def pre_compute_mean_std_for_TS(ta, m):
    na = len(ta)
    sum_t = np.zeros(na - m)
    sum_t2 = np.zeros(na - m)

    # Compute the stats for t
    cumulative_sum_t = np.cumsum(ta)
    cumulative_sum_t2 = np.cumsum(np.power(ta, 2))
    for i in range(na - m):
        sum_t[i] = cumulative_sum_t[i + m] - cumulative_sum_t[i]
        sum_t2[i] = cumulative_sum_t2[i + m] - cumulative_sum_t2[i]
    mean_t = np.divide(sum_t, m)
    mean_t2 = np.divide(sum_t2, m)
    mean_t_p2 = np.power(mean_t, 2)
    sigma_t2 = np.subtract(mean_t2, mean_t_p2)
    sigma_t = np.sqrt(sigma_t2)
    return sum_t, sum_t2, mean_t, mean_t2, mean_t_p2, sigma_t, sigma_t2


def pre_compute_mean_std_for_TS_stomp(ta, m):
    na = len(ta)
    # Compute the stats for t
    cumulative_sum_t = np.cumsum(ta)
    cumulative_sum_t2 = np.cumsum(np.power(ta, 2))
    sum_t = (cumulative_sum_t[m - 1:na] - np.concatenate(([0], cumulative_sum_t[0:na - m])))
    sum_t2 = (cumulative_sum_t2[m - 1:na] - np.concatenate(([0], cumulative_sum_t2[0:na - m])))
    mean_t = np.divide(sum_t, m)
    mean_t2 = np.divide(sum_t2, m)
    mean_t_p2 = np.power(mean_t, 2)
    sigma_t2 = np.subtract(mean_t2, mean_t_p2)
    sigma_t = np.sqrt(sigma_t2)
    return sum_t, sum_t2, mean_t, mean_t2, mean_t_p2, sigma_t, sigma_t2


# MUEEN’S ALGORITHM FOR SIMILARITY SEARCH (MASS)
def mass(Q, T, a, meanT, sigmaT):
    # Z-Normalisation
    if np.std(Q) != 0:
        Q = (Q - np.mean(Q)) / np.std(Q)
    QT = sliding_dot_product(Q, T)
    sumQ, sumQ2 = compute_mean_std_for_query(Q)
    return calculate_distance_profile(Q, T, QT, a, sumQ, sumQ2, meanT, sigmaT)


def element_wise_min(Pab, Iab, D, idx, ignore_trivial, m):
    for i in range(0, len(D)):
        if not ignore_trivial or (
                np.abs(idx - i) > m / 2.0):  # if it's a self-join, ignore trivial matches in [-m/2,m/2]
            if D[i] < Pab[i]:
                Pab[i] = D[i]
                Iab[i] = idx
    return Pab, Iab


def stamp(Ta, Tb, m):
    """
    Compute the Matrix Profile between time-series Ta and Tb.
    If Ta==Tb, the operation is a self-join and trivial matches are ignored.

    :param Ta: time-series, np.array
    :param Tb: time-series, np.array
    :param m: subsequence length
    :return: Matrix Profile, Nearest-Neighbor indexes
    """
    nb = len(Tb)
    na = len(Ta)
    Pab = np.ones(na - m) * np.inf
    Iab = np.zeros(na - m)
    idxes = np.arange(nb - m + 1)

    sumT, sumT2, meanT, meanT_2, meanTP2, sigmaT, sigmaT2 = pre_compute_mean_std_for_TS(Ta, m)

    a = np.zeros(na - m)
    for i in range(0, na - m):
        a[i] = (sumT2[i] - 2 * sumT[i] * meanT[i] + m * meanTP2[i]) / sigmaT2[i]

    ignore_trivial = np.atleast_1d(Ta == Tb).all()
    for idx in idxes:
        D = mass(Tb[idx: idx + m], Ta, a, meanT, sigmaT)
        if (ignore_trivial):
            # ignore trivial minimum and  maximum
            minIdx = int(np.maximum(idx - m / 2.0, 0))
            maxIdx = int(np.minimum(idx + m / 2.0, len(D)))
            D[minIdx:maxIdx:1] = np.inf

        Iab[Pab > D] = i
        Pab = np.minimum(Pab, D)
    return Pab, Iab


def stomp(T, m):
    """
    Compute the Matrix Profile with self join for T
    :param T: time-series, np.array
    :param Tb: time-series, np.array
    :param m: subsequence length
    :return: Matrix Profile, Nearest-Neighbor indexes
    """
    epsilon = 1e-10

    n = len(T)

    seq_l = n - m
    _, _, meanT, _, _, sigmaT, _ = pre_compute_mean_std_for_TS_stomp(T, m)

    Pab = np.full(seq_l + 1, np.inf)
    Iab = np.zeros(n - m + 1)
    ignore_trivial = True
    for idx in range(0, seq_l):
        # There's somthing with normalization
        Q_std = sigmaT[idx] if sigmaT[idx] > epsilon else epsilon
        if idx == 0:
            QT = sliding_dot_product_stomp(T[0:m], T).real
            QT_first = np.copy(QT)
        else:
            QT[1:] = QT[0:-1] - (T[0:seq_l] * T[idx - 1]) + (T[m:n] * T[idx + m - 1])
            QT[0] = QT_first[idx]

        # Calculate distance profile
        D = (2 * (m - (QT - m * meanT * meanT[idx]) / (Q_std * sigmaT)))
        D[D < epsilon] = 0
        if (ignore_trivial):
            # ignore trivial minimum and  maximum
            minIdx = int(np.maximum(idx - m / 2.0, 0))
            maxIdx = int(np.minimum(idx + m / 2.0, len(D)))
            D[minIdx:maxIdx:1] = np.inf

        Iab[Pab > D] = idx
        np.minimum(Pab, D, Pab)

    np.sqrt(Pab, Pab)
    return Pab, Iab


# Quick Test
# def test_stomp(Ta, m):
#     start_time = time.time()
#
#     Pab, Iab = stomp(Ta, m)
#     print("--- %s seconds ---" % (time.time() - start_time))
#     plot_motif(Ta, Pab, Iab, m)
#     return Pab, Iab


# Quick Test
# def test_stamp(Ta, Tb, m):
#     start_time = time.time()
#
#     Pab, Iab = stamp(Ta, Tb, m)
#     print("--- %s seconds ---" % (time.time() - start_time))
#
#     plot_discord(Ta, Pab, Iab, m, )
#     return Pab, Iab


def plot_motif(Ta, values, indexes, m):
    from matplotlib import gridspec
    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(Ta, linestyle='--', alpha=0.5)
    plt.xlim((0, len(Ta)))

    print(np.argmax(values))

    plt.plot(range(np.argmin(values), np.argmin(values) + m), Ta[np.argmin(values):np.argmin(values) + m], c='g',
             label='Top Motif')
    plt.plot(range(np.argmax(values), np.argmax(values) + m), Ta[np.argmax(values):np.argmax(values) + m], c='r',
             label='Top Discord')

    plt.legend(loc='best')
    plt.title('Time-Series')

    plt.subplot(212)
    plt.title('Matrix Profile')
    plt.plot(range(0, len(values)), values, '#ff5722')
    plt.plot(np.argmax(values), np.max(values), marker='x', c='r', ms=10)
    plt.plot(np.argmin(values), np.min(values), marker='^', c='g', ms=10)

    plt.xlim((0, len(Ta)))
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()


def plot_discord(Ta, Tb, values, indexes, m):
    from matplotlib import gridspec
    plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[int(len(Ta) / len(Tb)), 1])

    plt.subplot(gs[0])
    plt.plot(Ta, linestyle='--')
    plt.xlim((0, len(Ta)))

    plt.plot(range(np.argmin(values), np.argmin(values) + m), Ta[np.argmin(values):np.argmin(values) + m], c='g',
             label='Best Match')
    plt.legend(loc='best')
    plt.title('Time-Series')
    plt.ylim((-3, 3))

    plt.subplot(gs[1])
    plt.plot(Tb)

    plt.title('Query')
    plt.xlim((0, len(Tb)))
    plt.ylim((-3, 3))

    plt.figure()
    plt.title('Matrix Profile')
    plt.plot(range(0, len(values)), values, '#ff5722')
    plt.plot(np.argmax(values), np.max(values), marker='x', c='r', ms=10)
    plt.plot(np.argmin(values), np.min(values), marker='^', c='g', ms=10)

    plt.xlim((0, len(Ta)))
    plt.xlabel('Index')
    plt.ylabel('Value')

    plt.show()


def plot_match(Ta, Tb, values, indexes, m):
    from matplotlib import gridspec
    plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[int(len(Ta) / len(Tb)), 1])

    plt.subplot(gs[0])
    plt.plot(Ta, linestyle='--')
    plt.xlim((0, len(Ta)))

    print(np.argmax(values))

    plt.plot(range(np.argmin(values), np.argmin(values) + m), Ta[np.argmin(values):np.argmin(values) + m], c='g',
             label='Best Match')
    plt.legend(loc='best')
    plt.title('Time-Series')
    plt.ylim((-3, 3))

    plt.subplot(gs[1])
    plt.plot(Tb)

    plt.title('Query')
    plt.xlim((0, len(Tb)))
    plt.ylim((-3, 3))

    plt.figure()
    plt.title('Matrix Profile')
    plt.plot(range(0, len(values)), values, '#ff5722')
    plt.plot(np.argmax(values), np.max(values), marker='x', c='r', ms=10)
    plt.plot(np.argmin(values), np.min(values), marker='^', c='g', ms=10)

    plt.xlim((0, len(Ta)))
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()

def RunModel(_file_name, _choice, _element_num):
    pattern_size = 5
    if _choice == 1:
        abnormal_data, abnormal_label = ReadGDDataset(_file_name)
    if _choice == 2:
        abnormal_data, abnormal_label = ReadHSSDataset(_file_name)
    if _choice == 3:
        abnormal_data, abnormal_label = ReadS5Dataset(_file_name)
    if _choice == 4:
        abnormal_data, abnormal_label = ReadNABDataset(_file_name)
    if _choice == 5:
        abnormal_data, abnormal_label = Read2DDataset(_file_name)
    if _choice == 6:
        abnormal_data, abnormal_label = ReadUAHDataset(_file_name)
    if _choice == 7:
        abnormal_data, abnormal_label = ReadECGDataset(_file_name)
    ts = abnormal_data.flatten()
    query = abnormal_data.flatten()
    Pab, Iab = stamp(ts, query, pattern_size * _element_num)
    # plot_discord(ts, query, Pab, Iab, pattern_size * elem_num)
    final_zscore = Z_Score(np.sum(np.nan_to_num(Pab).reshape([-1, _element_num]), axis=1))
    y_pred = CreateLabelBasedOnZscore(final_zscore, 3, True)
    precision, recall, f1 = CalculatePrecisionRecallF1Metrics(abnormal_label[:-pattern_size], y_pred)
    # PrintPrecisionRecallF1Metrics(precision, recall, f1)
    fpr, tpr, roc_auc = CalculateROCAUCMetrics(abnormal_label[:-pattern_size], np.sum(np.nan_to_num(Pab).reshape([-1, _element_num]), axis=1))
    # print('roc_auc=' + str(roc_auc))
    precision_curve, recall_curve, average_precision = CalculatePrecisionRecallCurve(abnormal_label[:-pattern_size], np.sum(np.nan_to_num(Pab).reshape([-1, _element_num]), axis=1))
    # print('pr_auc=' + str(average_precision))
    cks = CalculateCohenKappaMetrics(abnormal_label[:-pattern_size], y_pred)
    # print('cohen_kappa=' + str(cks))

    return precision, recall, f1, roc_auc, average_precision, cks

if __name__ == '__main__':
    try:
        sys.argv[1]
    except IndexError:
        for n in range(1, 7):
            dataset = n
            if dataset == 1:
                file_name = './GD/data/Genesis_AnomalyLabels.csv'
                print(file_name)
                precision, recall, f1, roc_auc, pr_auc, cks = RunModel(file_name, dataset)
                print('avg_precision=' + str(precision))
                print('avg_recall=' + str(recall))
                print('avg_f1=' + str(f1))
                print('avg_roc_auc=' + str(roc_auc))
                print('avg_pr_auc=' + str(pr_auc))
                print('avg_cks=' + str(cks))
            
            if dataset == 2:
                file_name = './HSS/data/HRSS_anomalous_standard.csv'
                print(file_name)
                precision, recall, f1, roc_auc, pr_auc, cks = RunModel(file_name, dataset)
                print('avg_precision=' + str(precision))
                print('avg_recall=' + str(recall))
                print('avg_f1=' + str(f1))
                print('avg_roc_auc=' + str(roc_auc))
                print('avg_pr_auc=' + str(pr_auc))
                print('avg_cks=' + str(cks))

            if dataset == 3:
                for root, dirs, _ in os.walk('./YAHOO/data'):
                    for dir in dirs:
                        k_partition = 10
                        s_precision = []
                        s_recall = []
                        s_f1 = []
                        s_roc_auc = []
                        s_pr_auc = []
                        s_cks = []
                        for _, _, files in os.walk(root + '/' + dir):
                            for file in files:
                                file_name = os.path.join(root, dir, file)
                                print(file_name)
                                precision, recall, f1, roc_auc, pr_auc, cks = RunModel(file_name, dataset, _element_num=1)
                                s_precision.append(precision)
                                s_recall.append(recall)
                                s_f1.append(f1)
                                s_roc_auc.append(roc_auc)
                                s_pr_auc.append(pr_auc)
                                s_cks.append(cks)
                        print('########################################')
                        avg_precision = CalculateAverageMetric(s_precision)
                        print('avg_precision=' + str(avg_precision))
                        avg_recall = CalculateAverageMetric(s_recall)
                        print('avg_recall=' + str(avg_recall))
                        avg_f1 = CalculateAverageMetric(s_f1)
                        print('avg_f1=' + str(avg_f1))
                        avg_roc_auc = CalculateAverageMetric(s_roc_auc)
                        print('avg_roc_auc=' + str(avg_roc_auc))
                        avg_pr_auc = CalculateAverageMetric(s_pr_auc)
                        print('avg_pr_auc=' + str(avg_pr_auc))
                        avg_cks = CalculateAverageMetric(s_cks)
                        print('avg_cks=' + str(avg_cks))
                        print('########################################')

            if dataset == 4:
                for root, dirs, _ in os.walk('./NAB/data'):
                    for dir in dirs:
                        k_partition = 10
                        s_precision = []
                        s_recall = []
                        s_f1 = []
                        s_roc_auc = []
                        s_pr_auc = []
                        s_cks = []
                        for _, _, files in os.walk(root + '/' + dir):
                            for file in files:
                                file_name = os.path.join(root, dir, file)
                                print(file_name)
                                precision, recall, f1, roc_auc, pr_auc, cks = RunModel(file_name, dataset, _element_num=1)
                                s_precision.append(precision)
                                s_recall.append(recall)
                                s_f1.append(f1)
                                s_roc_auc.append(roc_auc)
                                s_pr_auc.append(pr_auc)
                                s_cks.append(cks)
                        print('########################################')
                        avg_precision = CalculateAverageMetric(s_precision)
                        print('avg_precision=' + str(avg_precision))
                        avg_recall = CalculateAverageMetric(s_recall)
                        print('avg_recall=' + str(avg_recall))
                        avg_f1 = CalculateAverageMetric(s_f1)
                        print('avg_f1=' + str(avg_f1))
                        avg_roc_auc = CalculateAverageMetric(s_roc_auc)
                        print('avg_roc_auc=' + str(avg_roc_auc))
                        avg_pr_auc = CalculateAverageMetric(s_pr_auc)
                        print('avg_pr_auc=' + str(avg_pr_auc))
                        avg_cks = CalculateAverageMetric(s_cks)
                        print('avg_cks=' + str(avg_cks))
                        print('########################################')

            if dataset == 5:
                for root, dirs, _ in os.walk('./2D/test'):
                    for dir in dirs:
                        k_partition = 3
                        s_precision = []
                        s_recall = []
                        s_f1 = []
                        s_roc_auc = []
                        s_pr_auc = []
                        s_cks = []
                        for _, _, files in os.walk(root + '/' + dir):
                            for file in files:
                                file_name = os.path.join(root, dir, file)
                                print(file_name)
                                precision, recall, f1, roc_auc, pr_auc, cks = RunModel(file_name, dataset, _element_num=2)
                                s_precision.append(precision)
                                s_recall.append(recall)
                                s_f1.append(f1)
                                s_roc_auc.append(roc_auc)
                                s_pr_auc.append(pr_auc)
                                s_cks.append(cks)
                        print('########################################')
                        avg_precision = CalculateAverageMetric(s_precision)
                        print('avg_precision=' + str(avg_precision))
                        avg_recall = CalculateAverageMetric(s_recall)
                        print('avg_recall=' + str(avg_recall))
                        avg_f1 = CalculateAverageMetric(s_f1)
                        print('avg_f1=' + str(avg_f1))
                        avg_roc_auc = CalculateAverageMetric(s_roc_auc)
                        print('avg_roc_auc=' + str(avg_roc_auc))
                        avg_pr_auc = CalculateAverageMetric(s_pr_auc)
                        print('avg_pr_auc=' + str(avg_pr_auc))
                        avg_cks = CalculateAverageMetric(s_cks)
                        print('avg_cks=' + str(avg_cks))
                        print('########################################')

            if dataset == 6:
                s_precision = []
                s_recall = []
                s_f1 = []
                s_roc_auc = []
                s_pr_auc = []
                s_cks = []
                for root, dirs, files in os.walk('./UAH/'):
                    for dir in dirs:
                        folder_name = os.path.join(root, dir)
                        print(folder_name)
                        precision, recall, f1, roc_auc, pr_auc, cks = RunModel(folder_name, dataset, _element_num=4)
                        print('########################################')
                        print('precision=' + str(precision))
                        print('recall=' + str(recall))
                        print('f1=' + str(f1))
                        print('roc_auc=' + str(roc_auc))
                        print('pr_auc=' + str(pr_auc))
                        print('cks=' + str(cks))
                        print('########################################')
                        s_precision.append(precision)
                        s_recall.append(recall)
                        s_f1.append(f1)
                        s_roc_auc.append(roc_auc)
                        s_pr_auc.append(pr_auc)
                        s_cks.append(cks)
                print('########################################')
                avg_precision = CalculateAverageMetric(s_precision)
                print('avg_precision=' + str(avg_precision))
                avg_recall = CalculateAverageMetric(s_recall)
                print('avg_recall=' + str(avg_recall))
                avg_f1 = CalculateAverageMetric(s_f1)
                print('avg_f1=' + str(avg_f1))
                avg_roc_auc = CalculateAverageMetric(s_roc_auc)
                print('avg_roc_auc=' + str(avg_roc_auc))
                avg_pr_auc = CalculateAverageMetric(s_pr_auc)
                print('avg_pr_auc=' + str(avg_pr_auc))
                avg_cks = CalculateAverageMetric(s_cks)
                print('avg_cks=' + str(avg_cks))
                print('########################################')

            if dataset == 7:
                for root, dirs, _ in os.walk('./ECG/'):
                    for dir in dirs:
                        k_partition = 3
                        s_precision = []
                        s_recall = []
                        s_f1 = []
                        s_roc_auc = []
                        s_pr_auc = []
                        s_cks = []
                        for _, _, files in os.walk(root + '/' + dir):
                            for file in files:
                                file_name = os.path.join(root, dir, file)
                                print(file_name)
                                precision, recall, f1, roc_auc, pr_auc, cks = RunModel(file_name, dataset, _element_num=3)
                                s_precision.append(precision)
                                s_recall.append(recall)
                                s_f1.append(f1)
                                s_roc_auc.append(roc_auc)
                                s_pr_auc.append(pr_auc)
                                s_cks.append(cks)
                        print('########################################')
                        avg_precision = CalculateAverageMetric(s_precision)
                        print('avg_precision=' + str(avg_precision))
                        avg_recall = CalculateAverageMetric(s_recall)
                        print('avg_recall=' + str(avg_recall))
                        avg_f1 = CalculateAverageMetric(s_f1)
                        print('avg_f1=' + str(avg_f1))
                        avg_roc_auc = CalculateAverageMetric(s_roc_auc)
                        print('avg_roc_auc=' + str(avg_roc_auc))
                        avg_pr_auc = CalculateAverageMetric(s_pr_auc)
                        print('avg_pr_auc=' + str(avg_pr_auc))
                        avg_cks = CalculateAverageMetric(s_cks)
                        print('avg_cks=' + str(avg_cks))
                        print('########################################')
    else:
        dataset = int(sys.argv[1])
        if dataset == 1:
            file_name = './GD/data/Genesis_AnomalyLabels.csv'
            print(file_name)
            precision, recall, f1, roc_auc, pr_auc, cks = RunModel(file_name, dataset)
            print('avg_precision=' + str(precision))
            print('avg_recall=' + str(recall))
            print('avg_f1=' + str(f1))
            print('avg_roc_auc=' + str(roc_auc))
            print('avg_pr_auc=' + str(pr_auc))
            print('avg_cks=' + str(cks))
        
        if dataset == 2:
            file_name = './HSS/data/HRSS_anomalous_standard.csv'
            print(file_name)
            precision, recall, f1, roc_auc, pr_auc, cks = RunModel(file_name, dataset)
            print('avg_precision=' + str(precision))
            print('avg_recall=' + str(recall))
            print('avg_f1=' + str(f1))
            print('avg_roc_auc=' + str(roc_auc))
            print('avg_pr_auc=' + str(pr_auc))
            print('avg_cks=' + str(cks))

        if dataset == 3:
            for root, dirs, _ in os.walk('./YAHOO/data'):
                for dir in dirs:
                    k_partition = 10
                    s_precision = []
                    s_recall = []
                    s_f1 = []
                    s_roc_auc = []
                    s_pr_auc = []
                    s_cks = []
                    for _, _, files in os.walk(root + '/' + dir):
                        for file in files:
                            file_name = os.path.join(root, dir, file)
                            print(file_name)
                            precision, recall, f1, roc_auc, pr_auc, cks = RunModel(file_name, dataset, _element_num=1)
                            s_precision.append(precision)
                            s_recall.append(recall)
                            s_f1.append(f1)
                            s_roc_auc.append(roc_auc)
                            s_pr_auc.append(pr_auc)
                            s_cks.append(cks)
                    print('########################################')
                    avg_precision = CalculateAverageMetric(s_precision)
                    print('avg_precision=' + str(avg_precision))
                    avg_recall = CalculateAverageMetric(s_recall)
                    print('avg_recall=' + str(avg_recall))
                    avg_f1 = CalculateAverageMetric(s_f1)
                    print('avg_f1=' + str(avg_f1))
                    avg_roc_auc = CalculateAverageMetric(s_roc_auc)
                    print('avg_roc_auc=' + str(avg_roc_auc))
                    avg_pr_auc = CalculateAverageMetric(s_pr_auc)
                    print('avg_pr_auc=' + str(avg_pr_auc))
                    avg_cks = CalculateAverageMetric(s_cks)
                    print('avg_cks=' + str(avg_cks))
                    print('########################################')

        if dataset == 4:
            for root, dirs, _ in os.walk('./NAB/data'):
                for dir in dirs:
                    k_partition = 10
                    s_precision = []
                    s_recall = []
                    s_f1 = []
                    s_roc_auc = []
                    s_pr_auc = []
                    s_cks = []
                    for _, _, files in os.walk(root + '/' + dir):
                        for file in files:
                            file_name = os.path.join(root, dir, file)
                            print(file_name)
                            precision, recall, f1, roc_auc, pr_auc, cks = RunModel(file_name, dataset, _element_num=1)
                            s_precision.append(precision)
                            s_recall.append(recall)
                            s_f1.append(f1)
                            s_roc_auc.append(roc_auc)
                            s_pr_auc.append(pr_auc)
                            s_cks.append(cks)
                    print('########################################')
                    avg_precision = CalculateAverageMetric(s_precision)
                    print('avg_precision=' + str(avg_precision))
                    avg_recall = CalculateAverageMetric(s_recall)
                    print('avg_recall=' + str(avg_recall))
                    avg_f1 = CalculateAverageMetric(s_f1)
                    print('avg_f1=' + str(avg_f1))
                    avg_roc_auc = CalculateAverageMetric(s_roc_auc)
                    print('avg_roc_auc=' + str(avg_roc_auc))
                    avg_pr_auc = CalculateAverageMetric(s_pr_auc)
                    print('avg_pr_auc=' + str(avg_pr_auc))
                    avg_cks = CalculateAverageMetric(s_cks)
                    print('avg_cks=' + str(avg_cks))
                    print('########################################')

        if dataset == 5:
            for root, dirs, _ in os.walk('./2D/test'):
                for dir in dirs:
                    k_partition = 3
                    s_precision = []
                    s_recall = []
                    s_f1 = []
                    s_roc_auc = []
                    s_pr_auc = []
                    s_cks = []
                    for _, _, files in os.walk(root + '/' + dir):
                        for file in files:
                            file_name = os.path.join(root, dir, file)
                            print(file_name)
                            precision, recall, f1, roc_auc, pr_auc, cks = RunModel(file_name, dataset, _element_num=2)
                            s_precision.append(precision)
                            s_recall.append(recall)
                            s_f1.append(f1)
                            s_roc_auc.append(roc_auc)
                            s_pr_auc.append(pr_auc)
                            s_cks.append(cks)
                    print('########################################')
                    avg_precision = CalculateAverageMetric(s_precision)
                    print('avg_precision=' + str(avg_precision))
                    avg_recall = CalculateAverageMetric(s_recall)
                    print('avg_recall=' + str(avg_recall))
                    avg_f1 = CalculateAverageMetric(s_f1)
                    print('avg_f1=' + str(avg_f1))
                    avg_roc_auc = CalculateAverageMetric(s_roc_auc)
                    print('avg_roc_auc=' + str(avg_roc_auc))
                    avg_pr_auc = CalculateAverageMetric(s_pr_auc)
                    print('avg_pr_auc=' + str(avg_pr_auc))
                    avg_cks = CalculateAverageMetric(s_cks)
                    print('avg_cks=' + str(avg_cks))
                    print('########################################')


        if dataset == 6:
            s_precision = []
            s_recall = []
            s_f1 = []
            s_roc_auc = []
            s_pr_auc = []
            s_cks = []
            for root, dirs, files in os.walk('./UAH/'):
                for dir in dirs:
                    folder_name = os.path.join(root, dir)
                    print(folder_name)
                    precision, recall, f1, roc_auc, pr_auc, cks = RunModel(folder_name, dataset, _element_num=4)
                    print('########################################')
                    print('precision=' + str(precision))
                    print('recall=' + str(recall))
                    print('f1=' + str(f1))
                    print('roc_auc=' + str(roc_auc))
                    print('pr_auc=' + str(pr_auc))
                    print('cks=' + str(cks))
                    print('########################################')
                    s_precision.append(precision)
                    s_recall.append(recall)
                    s_f1.append(f1)
                    s_roc_auc.append(roc_auc)
                    s_pr_auc.append(pr_auc)
                    s_cks.append(cks)
            print('########################################')
            avg_precision = CalculateAverageMetric(s_precision)
            print('avg_precision=' + str(avg_precision))
            avg_recall = CalculateAverageMetric(s_recall)
            print('avg_recall=' + str(avg_recall))
            avg_f1 = CalculateAverageMetric(s_f1)
            print('avg_f1=' + str(avg_f1))
            avg_roc_auc = CalculateAverageMetric(s_roc_auc)
            print('avg_roc_auc=' + str(avg_roc_auc))
            avg_pr_auc = CalculateAverageMetric(s_pr_auc)
            print('avg_pr_auc=' + str(avg_pr_auc))
            avg_cks = CalculateAverageMetric(s_cks)
            print('avg_cks=' + str(avg_cks))
            print('########################################')

        if dataset == 7:
            for root, dirs, _ in os.walk('./ECG/'):
                for dir in dirs:
                    k_partition = 3
                    s_precision = []
                    s_recall = []
                    s_f1 = []
                    s_roc_auc = []
                    s_pr_auc = []
                    s_cks = []
                    for _, _, files in os.walk(root + '/' + dir):
                        for file in files:
                            file_name = os.path.join(root, dir, file)
                            print(file_name)
                            precision, recall, f1, roc_auc, pr_auc, cks = RunModel(file_name, dataset, _element_num=3)
                            s_precision.append(precision)
                            s_recall.append(recall)
                            s_f1.append(f1)
                            s_roc_auc.append(roc_auc)
                            s_pr_auc.append(pr_auc)
                            s_cks.append(cks)
                    print('########################################')
                    avg_precision = CalculateAverageMetric(s_precision)
                    print('avg_precision=' + str(avg_precision))
                    avg_recall = CalculateAverageMetric(s_recall)
                    print('avg_recall=' + str(avg_recall))
                    avg_f1 = CalculateAverageMetric(s_f1)
                    print('avg_f1=' + str(avg_f1))
                    avg_roc_auc = CalculateAverageMetric(s_roc_auc)
                    print('avg_roc_auc=' + str(avg_roc_auc))
                    avg_pr_auc = CalculateAverageMetric(s_pr_auc)
                    print('avg_pr_auc=' + str(avg_pr_auc))
                    avg_cks = CalculateAverageMetric(s_cks)
                    print('avg_cks=' + str(avg_cks))
                    print('########################################')
