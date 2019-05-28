import pathlib
import numpy as np
import pandas as pd
import json
import scipy.io
from numpy import linalg as LA
import matplotlib.pyplot as plt
import os
from os.path import join, getsize
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score, cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler


def ReadS5Dataset(_file_name, _normalize=True):
    abnormal = pd.read_csv(_file_name, header=0, index_col=None)
    abnormal_data = abnormal['value'].as_matrix()
    abnormal_label = abnormal['is_anomaly'].as_matrix()
    # Normal = 0, Abnormal = 1 => # Normal = 1, Abnormal = -1

    abnormal_data = np.expand_dims(abnormal_data, axis=1)
    abnormal_label = np.expand_dims(abnormal_label, axis=1)

    if _normalize==True:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)

    abnormal_label[abnormal_label == 1] = -1
    abnormal_label[abnormal_label == 0] = 1
    return abnormal_data, abnormal_label


def ReadNABDataset(_file_name, _normalize=True):
    with open('./NAB/labels/combined_windows.json') as data_file:
        json_label = json.load(data_file)
    abnormal = pd.read_csv(_file_name, header=0, index_col=0)
    abnormal['label'] = 1
    list_windows = json_label.get(os.path.basename(_file_name))
    for window in list_windows:
        start = window[0]
        end = window[1]
        abnormal.loc[start:end, 'label'] = -1

    abnormal_data = abnormal['value'].as_matrix()
    # abnormal_preprocessing_data = np.reshape(abnormal_preprocessing_data, (abnormal_preprocessing_data.shape[0], 1))
    abnormal_label = abnormal['label'].as_matrix()

    abnormal_data = np.expand_dims(abnormal_data, axis=1)
    abnormal_label = np.expand_dims(abnormal_label, axis=1)

    if _normalize==True:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)

    # Normal = 1, Abnormal = -1
    return abnormal_data, abnormal_label


def ReadUAHDataset(_file_folder, _normalize=True):
    def calculate_steering_angle(a):
        b = np.zeros(shape=(a.shape[0], 1))
        for i in range(a.size):
            if i == 0:
                b[i] = a[i]
            else:
                b[i] = (a[i] - a[i - 1])
                if b[i] >= 180:
                    b[i] = 360 - b[i]
                elif -180 < b[i] < 180:
                    b[i] = abs(b[i])
                elif b[i] <= -180:
                    b[i] = b[i] + 360
        return b

    def calculate_by_previous_element(_a):
        b = np.zeros(shape=(_a.shape[0], 1))
        for i in range(_a.size):
            if i == 0:
                b[i] = 0
            else:
                b[i] = (_a[i] - _a[i - 1])
        return b

    def ReadRawGPSDataset(_folder_name):
        dataset = np.loadtxt(fname=_folder_name + '/' + os.path.basename(_folder_name) + '_RAW_GPS.txt', delimiter=' ', usecols=(1, 7))
        return dataset

    def ReadTimestampAndLabelOfSemanticDataset(_folder_name):
        dataset = np.loadtxt(fname=_folder_name + '/' + os.path.basename(_folder_name) + '_SEMANTIC_ONLINE.txt', delimiter=' ', usecols=(0, 23, 24, 25))
        return dataset

    def PreprocessRawData(_raw_data):
        speed_array = _raw_data[:, 0]
        dir_array = _raw_data[:, 1]

        # calculate acceleration (diff of speed)
        acceleration_array = calculate_by_previous_element(speed_array)

        # calculate jerk (diff of acceleration)
        jerk_array = calculate_by_previous_element(acceleration_array)

        # calculate steering (diff of direction)
        steering_array = calculate_steering_angle(dir_array)

        add_acceleration = np.c_[speed_array, acceleration_array]
        add_jerk = np.c_[add_acceleration, jerk_array]
        add_steering = np.c_[add_jerk, steering_array]

        return add_steering

    def ComputeLabelForSemantic(semantic_online_data):
        label = np.zeros(semantic_online_data.shape[0])
        for i in range(semantic_online_data.shape[0]):
            if semantic_online_data[i][0] <= semantic_online_data[i][1] or semantic_online_data[i][0] <= \
                    semantic_online_data[i][2] or semantic_online_data[i][0] <= semantic_online_data[i][1] + \
                    semantic_online_data[i][2]:
                label[i] = -1
            else:
                label[i] = 1
        return label

    abnormal = ReadRawGPSDataset(_file_folder)
    abnormal_data = PreprocessRawData(abnormal)

    if _normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)

    abnormal_label = ReadTimestampAndLabelOfSemanticDataset(_file_folder)
    abnormal_label_data = ComputeLabelForSemantic(abnormal_label[:, [1, 2, 3]])

    return abnormal_data, abnormal_label_data


def Read2DDataset(_file_name, _normalize=True):
    abnormal = pd.read_csv(_file_name, header=None, index_col=None, skiprows=1, sep=' ')
    abnormal_data = abnormal.iloc[:, [0, 1]].as_matrix()
    abnormal_label = abnormal.iloc[:, 2].as_matrix()
    # Normal = 0, Abnormal = 1 => # Normal = 1, Abnormal = -1

    # abnormal_data = np.expand_dims(abnormal_data, axis=1)
    abnormal_label = np.expand_dims(abnormal_label, axis=1)

    if _normalize == True:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)

    abnormal_label[abnormal_label == 2] = -1
    abnormal_label[abnormal_label == 0] = 1
    return abnormal_data, abnormal_label


def ReadECGDataset(_file_name, _normalize=True):
    abnormal = pd.read_csv(_file_name, header=None, index_col=None, skiprows=0, sep=',')
    abnormal_data = abnormal.iloc[:, [0, 1, 2]].as_matrix()
    abnormal_label = abnormal.iloc[:, 3].as_matrix()
    # Normal = 0, Abnormal = 1 => # Normal = 1, Abnormal = -1

    # abnormal_data = np.expand_dims(abnormal_data, axis=1)
    abnormal_label = np.expand_dims(abnormal_label, axis=1)

    if _normalize == True:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)

    abnormal_label[abnormal_label == 1] = -1
    abnormal_label[abnormal_label == 0] = 1
    return abnormal_data, abnormal_label



def ReadGDDataset(_file_name, _normalize=True):
    abnormal = pd.read_csv(_file_name, header=0, index_col=0)
    abnormal_data = abnormal[['MotorData.ActCurrent', 'MotorData.ActPosition', 'MotorData.ActSpeed', 'MotorData.IsAcceleration',
                             'MotorData.IsForce', 'MotorData.Motor_Pos1reached', 'MotorData.Motor_Pos2reached', 'MotorData.Motor_Pos3reached',
                             'MotorData.Motor_Pos4reached', 'NVL_Recv_Ind.GL_Metall', 'NVL_Recv_Ind.GL_NonMetall',
                             'NVL_Recv_Storage.GL_I_ProcessStarted', 'NVL_Recv_Storage.GL_I_Slider_IN', 'NVL_Recv_Storage.GL_I_Slider_OUT',
                             'NVL_Recv_Storage.GL_LightBarrier', 'NVL_Send_Storage.ActivateStorage', 'PLC_PRG.Gripper', 'PLC_PRG.MaterialIsMetal']].as_matrix()
    if _normalize==True:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)

    abnormal_label = abnormal['Label'].as_matrix()
    # Normal = 0, Abnormal = 2 => # Normal = 1, Abnormal = -1
    abnormal_label[abnormal_label != 0] = -1
    abnormal_label[abnormal_label == 0] = 1
    return abnormal_data, abnormal_label


def ReadHSSDataset(_file_name, _normalize=True):
    abnormal = pd.read_csv(_file_name, header=0, index_col=0)
    abnormal_data = abnormal[['I_w_BLO_Weg', 'O_w_BLO_power', 'O_w_BLO_voltage', 'I_w_BHL_Weg', 'O_w_BHL_power', 'O_w_BHL_voltage',
                              'I_w_BHR_Weg', 'O_w_BHR_power', 'O_w_BHR_voltage', 'I_w_BRU_Weg', 'O_w_BRU_power', 'O_w_BRU_voltage',
                              'I_w_HR_Weg', 'O_w_HR_power', 'O_w_HR_voltage', 'I_w_HL_Weg', 'O_w_HL_power', 'O_w_HL_voltage']].as_matrix()
    if _normalize==True:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)

    abnormal_label = abnormal['Labels'].as_matrix()
    # Normal = 0, Abnormal = 1 => # Normal = 1, Abnormal = -1
    abnormal_label[abnormal_label != 0] = -1
    abnormal_label[abnormal_label == 0] = 1
    return abnormal_data, abnormal_label


def CalculatePrecisionRecallF1Metrics(_abnormal_label, _y_pred):
    precision = precision_score(_abnormal_label, _y_pred)
    recall = recall_score(_abnormal_label, _y_pred)
    f1 = f1_score(_abnormal_label, _y_pred)
    return precision, recall, f1


def CreateTopKLabelBasedOnReconstructionError(_error, _k):
    label = np.full(_error.shape[0], 1)
    outlier_indices = _error.argsort()[-_k:][::-1]
    for i in outlier_indices:
        label[i] = -1
    return label, outlier_indices


def CalculatePrecisionAtK(_abnormal_label, _score, _k, _type):
    y_pred_at_k = np.full(_k, -1)
    if _type == 1:  # Local Outlier Factor & Auto-Encoder Type
        # _score[_score > 2.2] = 1
        outlier_indices = _score.argsort()[-_k:][::-1]
    if _type == 2:  # Isolation Forest & One-class SVM Type
        outlier_indices = _score.argsort()[:_k]
    abnormal_at_k = []
    for i in outlier_indices:
        abnormal_at_k.append(_abnormal_label[i])
    abnormal_at_k = np.asarray(abnormal_at_k)
    precision_at_k = precision_score(abnormal_at_k, y_pred_at_k)
    return precision_at_k


def CalculateROCAUCMetrics(_abnormal_label, _score):
    fpr, tpr, _ = roc_curve(_abnormal_label, _score)
    roc_auc = auc(np.nan_to_num(fpr), np.nan_to_num(tpr))
    if roc_auc < 0.5:
        roc_auc = 1 - roc_auc
    return fpr, tpr, roc_auc


def CalculateCohenKappaMetrics(_abnormal_label, _y_pred):
    cks = cohen_kappa_score(_abnormal_label, _y_pred)
    if cks < 0:
        cks = 0
    return cks


def CalculatePrecisionRecallCurve(_abnormal_label, _score):
    precision_curve, recall_curve, _ = precision_recall_curve(_abnormal_label, _score)
    average_precision = average_precision_score(_abnormal_label, _score)
    if average_precision < 0.5:
        average_precision = 1 - average_precision
    return precision_curve, recall_curve, average_precision


def CalculateFinalAnomalyScore(_ensemble_score):
    final_score = np.median(_ensemble_score, axis=0)
    return final_score


def PrintPrecisionRecallF1Metrics(_precision, _recall, _f1):
    print('precision=' + str(_precision))
    print('recall=' + str(_recall))
    print('f1=' + str(_f1))


def CalculateAverageMetric(_sum_of_score):
    '''
    Calculate average score of a set of multiple dataset
    :param _sum_of_score: Python List [] of score
    :return: average score
    '''
    average_score = sum(_sum_of_score) / float(len(_sum_of_score))
    return average_score


def PrintROCAUCMetrics(_fpr, _tpr, _roc_auc):
    print('fpr=' + str(_fpr))
    print('tpr=' + str(_tpr))
    print('roc_auc' + str(_roc_auc))


def SquareErrorDataPoints(_input, _output):
    input = np.squeeze(_input, axis=0)
    output = np.squeeze(_output, axis=0)
    # Caculate error
    error = np.square(input - output)
    error = np.sum(error, axis=1)
    return error


def Z_Score(_error):
    mu = np.nanmean(_error)
    gamma = np.nanstd(_error)
    zscore = (_error - mu)/gamma
    return zscore


def PlotResult(_values):
    plt.plot(_values)
    plt.show()


def CreateLabelBasedOnReconstructionError(_error, _percent_of_outlier):
    label = np.full(_error.shape[0], 1)
    number_of_outlier = _error.shape[0] * _percent_of_outlier
    outlier_indices = _error.argsort()[-number_of_outlier:][::-1]
    for i in outlier_indices:
        label[i] = -1
    return label


def CreateLabelBasedOnZscore(_zscore, _threshold, _sign=False):
    label = np.full(_zscore.shape[0], 1)
    if not _sign:
        label[_zscore > _threshold] = -1
        label[_zscore < -_threshold] = -1
    else:
        label[_zscore > _threshold] = -1
    # label[abs(_zscore) > abs(_threshold)] = -1
    return label


def PartitionTimeSeriesKPart(_timeseries, _label, _part_number=10):
    splitted_data = np.array_split(_timeseries, _part_number, axis=1)
    splitted_label = np.array_split(_label, _part_number, axis=0)
    return splitted_data, splitted_label


def PlotROCAUC(_fpr, _tpr, _roc_auc):
    plt.figure(1)
    lw = 1.5
    plt.plot(_fpr, _tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % _roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def PlotPrecisionRecallCurve(_precision, _recall, _average_precision):
    plt.figure(2)
    lw = 2
    plt.step(_recall, _precision, color='darkorange', lw=lw, alpha=1, where='post', label='PR curve (area = %0.2f)' % _average_precision)
    # plt.fill_between(_recall, _precision, step='post', alpha=0.2, color='b')
    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    # plt.legend('AP={0:0.2f}'.format(_average_precision))
    plt.legend(loc="lower right")
    plt.show()



# ReadS5Dataset('./YAHOO/data/A1Benchmark/real_1.csv')
# ReadGDDataset('./GD/data/Genesis_AnomalyLabels.csv')
# ReadHSSDataset('./HSS/data/HRSS_anomalous_standard.csv')