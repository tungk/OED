import pathlib
import sys

from sklearn.neighbors import lof
from utils import *


def LOFModel(_abnormal_data, _neighbor=5, _job=2, _contamination=0.05, _metric='euclidean'):
    clf = lof.LocalOutlierFactor(n_neighbors=_neighbor, n_jobs=_job, metric=_metric)
    y_pred = clf.fit_predict(_abnormal_data)
    score = clf.negative_outlier_factor_
    return score, y_pred


def RunModel(_file_name, _choice, _neighbor=5, _job=2, _contamination=0.1, _metric='euclidean'):
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
    score, y_pred = LOFModel(abnormal_data, _neighbor, _job, _metric)
    score_pred_label = np.c_[score, y_pred, abnormal_label]
    np.savetxt('./saved_result/' + pathlib.Path(_file_name).parts[0] + '/lof_' + os.path.basename(_file_name) + '_score.txt', score_pred_label, delimiter=',')

    x = abnormal_label[np.where(abnormal_label == -1)]
    y = y_pred[np.where(y_pred == -1)]

    precision, recall, f1 = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
    # PrintPrecisionRecallF1Metrics(precision, recall, f1)

    # k_number = [20, 40, 60, 80, 100]
    # for k in k_number:
    #     precision_at_k = CalculatePrecisionAtK(abnormal_label, -score, k, _type=1)
    #     print('precision at ' + str(k) + '=' + str(precision_at_k))

    fpr, tpr, roc_auc = CalculateROCAUCMetrics(abnormal_label, score)
    # PlotROCAUC(fpr, tpr, roc_auc)
    # print('roc_auc=' + str(roc_auc))
    precision_curve, recall_curve, average_precision = CalculatePrecisionRecallCurve(abnormal_label, score)
    # PlotPrecisionRecallCurve(precision_curve, recall_curve, average_precision)
    # print('pr_auc=' + str(average_precision))
    cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
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
                                precision, recall, f1, roc_auc, pr_auc, cks = RunModel(file_name, dataset)
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
                                precision, recall, f1, roc_auc, pr_auc, cks = RunModel(file_name, dataset)
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
                                precision, recall, f1, roc_auc, pr_auc, cks = RunModel(file_name, dataset)
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
                    for file in files:
                        file_name = os.path.join(root, file)
                        print(file_name)
                        precision, recall, f1, roc_auc, pr_auc, cks = RunModel(file_name, dataset)
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
                                precision, recall, f1, roc_auc, pr_auc, cks = RunModel(file_name, dataset)
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
                            precision, recall, f1, roc_auc, pr_auc, cks = RunModel(file_name, dataset)
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
                            precision, recall, f1, roc_auc, pr_auc, cks = RunModel(file_name, dataset)
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
                            precision, recall, f1, roc_auc, pr_auc, cks = RunModel(file_name, dataset)
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
                    precision, recall, f1, roc_auc, pr_auc, cks = RunModel(folder_name, dataset)
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
                            precision, recall, f1, roc_auc, pr_auc, cks = RunModel(file_name, dataset)
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