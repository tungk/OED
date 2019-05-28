import pathlib
import sys

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.rnn_cell_impl import _Linear, LSTMStateTuple
from tensorflow.python.ops import variable_scope as vs
from utils import *

def conv1d_relu(_x, _w, _b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv1d(_x, _w, stride=1, padding='SAME'), _b))

def conv1d_sigmoid(_x, _w, _b):
    return tf.nn.sigmoid(tf.nn.bias_add(tf.nn.conv1d(_x, _w, stride=1, padding='SAME'), _b))


def Model(_abnormal_data, _abnormal_label):
    g = tf.Graph()
    with g.as_default():
        # placeholder list
        p_input = tf.placeholder(tf.float32, shape=(batch_num, _abnormal_data.shape[1], _abnormal_data.shape[2]))

        # Weight and Bias for convolution encoding
        wc1_enc = tf.Variable(tf.random_normal([5, elem_num, 64]))
        bc1_enc = tf.Variable(tf.random_normal([64]))

        wc2_enc = tf.Variable(tf.random_normal([5, 64, 32]))
        bc2_enc = tf.Variable(tf.random_normal([32]))

        wc3_enc = tf.Variable(tf.random_normal([5, 32, 16]))
        bc3_enc = tf.Variable(tf.random_normal([16]))

        # Weight and Bias for convolution decoding
        wc1_dec = tf.Variable(tf.random_normal([5, 16, 32]))
        bc1_dec = tf.Variable(tf.random_normal([32]))

        wc2_dec = tf.Variable(tf.random_normal([5, 32, 64]))
        bc2_dec = tf.Variable(tf.random_normal([64]))

        wc3_dec = tf.Variable(tf.random_normal([5, 64, elem_num]))
        bc3_dec = tf.Variable(tf.random_normal([elem_num]))

        # with tf.device('/device:GPU:0'):
        with tf.variable_scope('encoder'):
            # Conv 1st layer
            conv1_enc = conv1d_relu(p_input, wc1_enc, bc1_enc)

            # Conv 2nd layer
            conv2_enc = conv1d_relu(conv1_enc, wc2_enc, bc2_enc)

            # Conv 3rd layer
            conv3_enc = conv1d_relu(conv2_enc, wc3_enc, bc3_enc)

        # with tf.device('/device:GPU:1'):
        with tf.variable_scope('decoder'):
            # Conv 1st layer
            conv1_dec = conv1d_relu(conv3_enc, wc1_dec, bc1_dec)

            # Conv 2nd layer
            conv2_dec = conv1d_relu(conv1_dec, wc2_dec, bc2_dec)

            # Conv 3rd layer
            dec_outputs = conv1d_sigmoid(conv2_dec, wc3_dec, bc3_dec)

        loss = tf.reduce_mean(tf.square(p_input - dec_outputs))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
    return g, p_input, dec_outputs, loss, optimizer, saver

def RunModel(_abnormal_data, _abnormal_label):
    graph, p_input, dec_outputs, loss, optimizer, saver = Model(_abnormal_data, _abnormal_label)
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(iteration):
            """Random sequences.
              Every sequence has size batch_num * step_num * elem_num 
              Each step number increases 1 by 1.
              An initial number of each sequence is in the range from 0 to 19.
              (ex. [8. 9. 10. 11. 12. 13. 14. 15])
            """

            (loss_val, _) = sess.run([loss, optimizer], {p_input: _abnormal_data})
            # print('iter %d:' % (i + 1), loss_val)

        if not partition:
            save_path = saver.save(sess, './saved_model/' + pathlib.Path(file_name).parts[0] + '/cnn_seq2seq_' + os.path.basename(file_name) + '.ckpt')
            print("Model saved in path: %s" % save_path)

        (input_, output_) = sess.run([p_input, dec_outputs], {p_input: _abnormal_data})
        error = SquareErrorDataPoints(input_, output_)

        # np.savetxt('./saved_result/' + pathlib.Path(file_name).parts[0] + '/cnn_seq2seq_' + os.path.basename(file_name) + '_error.txt', error, delimiter=',')  # X is an array
        zscore = Z_Score(error)
        # np.savetxt('./saved_result/' + pathlib.Path(file_name).parts[0] + '/cnn_seq2seq_' + os.path.basename(file_name) + '_zscore.txt', zscore, delimiter=',')  # X is an array

        y_pred = CreateLabelBasedOnZscore(zscore, 3)

        if not partition:
            score_pred_label = np.c_[error, y_pred, _abnormal_label]
            np.savetxt('./saved_result/' + pathlib.Path(file_name).parts[0] + '/cnn_seq2seq_' + os.path.basename(file_name) + '_score.txt', score_pred_label, delimiter=',')  # X is an array

        p, r, f = CalculatePrecisionRecallF1Metrics(_abnormal_label, y_pred)
        if not partition:
            PrintPrecisionRecallF1Metrics(p, r, f)

        # k_number = [20, 40, 60, 80, 100]
        # for k in k_number:
        #     precision_at_k = CalculatePrecisionAtK(_abnormal_label, error, k, _type=1)
        #     print('precision at ' + str(k) + '=' + str(precision_at_k))


        fpr, tpr, average_roc_auc = CalculateROCAUCMetrics(_abnormal_label, error)
        # PlotROCAUC(fpr, tpr, roc_auc)
        if not partition:
            print('roc_auc=' + str(average_roc_auc))

        precision_curve, recall_curve, average_precision = CalculatePrecisionRecallCurve(_abnormal_label, error)
        # PlotPrecisionRecallCurve(precision_curve, recall_curve, average_precision)
        if not partition:
            print('pr_auc=' + str(average_precision))

        cks = CalculateCohenKappaMetrics(_abnormal_label, y_pred)
        if not partition:
            print('cks=' + str(cks))

        return error, p, r, f, average_roc_auc, average_precision, cks


if __name__ == '__main__':
    batch_num = 1
    hidden_num = 4
    # step_num = 8
    iteration = 30
    ensemble_space = 10
    learning_rate = 1e-3
    multivariate = True

    partition = True
    save_model = False
    try:
        sys.argv[1]
    except IndexError:
        for n in range(1, 7):
            # file name parameter
            dataset = n
            if dataset == 1:
                file_name = './GD/data/Genesis_AnomalyLabels.csv'
                print(file_name)
                k_partition = 40
                abnormal_data, abnormal_label = ReadGDDataset(file_name)
                elem_num = 18
                if multivariate:
                    abnormal_data = np.expand_dims(abnormal_data, axis=0)
                if partition:
                    splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data, abnormal_label, _part_number=k_partition)
                    final_error = []
                    for i in range(k_partition):
                        error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(splitted_data[i], splitted_label[i])
                        final_error.append(error_partition)
                    # print('-----------------------------------------')
                    final_error = np.concatenate(final_error).ravel()
                    final_zscore = Z_Score(final_error)
                    y_pred = CreateLabelBasedOnZscore(final_zscore, 2.5)
                    final_p, final_r, final_f = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
                    PrintPrecisionRecallF1Metrics(final_p, final_r, final_f)
                    final_fpr, final_tpr, final_average_roc_auc = CalculateROCAUCMetrics(abnormal_label, final_error)
                    print('roc_auc=' + str(final_average_roc_auc))
                    final_precision_curve, final_recall_curve, final_average_precision = CalculatePrecisionRecallCurve(abnormal_label, final_error)
                    print('pr_auc=' + str(final_average_precision))
                    cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                    print('cohen_kappa=' + str(cks))
                else:
                    error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label)
            
            if dataset == 2:
                file_name = './HSS/data/HRSS_anomalous_standard.csv'
                print(file_name)
                k_partition = 80
                abnormal_data, abnormal_label = ReadHSSDataset(file_name)
                elem_num = 18
                if multivariate:
                    abnormal_data = np.expand_dims(abnormal_data, axis=0)
                if partition:
                    splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data, abnormal_label, _part_number=k_partition)
                    final_error = []
                    for i in range(k_partition):
                        error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(splitted_data[i], splitted_label[i])
                        final_error.append(error_partition)
                    # print('-----------------------------------------')
                    final_error = np.concatenate(final_error).ravel()
                    final_zscore = Z_Score(final_error)
                    y_pred = CreateLabelBasedOnZscore(final_zscore, 3)
                    final_p, final_r, final_f = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
                    PrintPrecisionRecallF1Metrics(final_p, final_r, final_f)
                    final_fpr, final_tpr, final_average_roc_auc = CalculateROCAUCMetrics(abnormal_label, final_error)
                    print('roc_auc=' + str(final_average_roc_auc))
                    final_precision_curve, final_recall_curve, final_average_precision = CalculatePrecisionRecallCurve(abnormal_label, final_error)
                    print('pr_auc=' + str(final_average_precision))
                    cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                    print('cohen_kappa=' + str(cks))
                else:
                    error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label)

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
                                abnormal_data, abnormal_label = ReadS5Dataset(file_name)
                                elem_num = 1
                                if multivariate:
                                    abnormal_data = np.expand_dims(abnormal_data, axis=0)
                                if partition:
                                    splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data, abnormal_label, _part_number=k_partition)
                                    final_error = []
                                    for i in range(k_partition):
                                        error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, pr_cks = RunModel(splitted_data[i], splitted_label[i])
                                        final_error.append(error_partition)
                                    # print('-----------------------------------------')
                                    final_error = np.concatenate(final_error).ravel()
                                    final_zscore = Z_Score(final_error)
                                    y_pred = CreateLabelBasedOnZscore(final_zscore, 3)
                                    final_p, final_r, final_f = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
                                    PrintPrecisionRecallF1Metrics(final_p, final_r, final_f)
                                    final_fpr, final_tpr, final_average_roc_auc = CalculateROCAUCMetrics(abnormal_label, final_error)
                                    print('roc_auc=' + str(final_average_roc_auc))
                                    final_precision_curve, final_recall_curve, final_average_precision = CalculatePrecisionRecallCurve(abnormal_label, final_error)
                                    print('pr_auc=' + str(final_average_precision))
                                    cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                                    print('cohen_kappa=' + str(cks))
                                else:
                                    error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label)

                                s_precision.append(final_p)
                                s_recall.append(final_r)
                                s_f1.append(final_f)
                                s_roc_auc.append(final_average_roc_auc)
                                s_pr_auc.append(final_average_precision)
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
                                abnormal_data, abnormal_label = ReadNABDataset(file_name)
                                elem_num = 1
                                if multivariate:
                                    abnormal_data = np.expand_dims(abnormal_data, axis=0)
                                if partition:
                                    splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data, abnormal_label, _part_number=k_partition)
                                    final_error = []
                                    for i in range(k_partition):
                                        error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, pr_cks = RunModel(splitted_data[i], splitted_label[i])
                                        final_error.append(error_partition)
                                    # print('-----------------------------------------')
                                    final_error = np.concatenate(final_error).ravel()
                                    final_zscore = Z_Score(final_error)
                                    y_pred = CreateLabelBasedOnZscore(final_zscore, 3)
                                    final_p, final_r, final_f = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
                                    PrintPrecisionRecallF1Metrics(final_p, final_r, final_f)
                                    final_fpr, final_tpr, final_average_roc_auc = CalculateROCAUCMetrics(abnormal_label, final_error)
                                    print('roc_auc=' + str(final_average_roc_auc))
                                    final_precision_curve, final_recall_curve, final_average_precision = CalculatePrecisionRecallCurve(abnormal_label, final_error)
                                    print('pr_auc=' + str(final_average_precision))
                                    cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                                    print('cohen_kappa=' + str(cks))
                                else:
                                    error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label)

                                s_precision.append(final_p)
                                s_recall.append(final_r)
                                s_f1.append(final_f)
                                s_roc_auc.append(final_average_roc_auc)
                                s_pr_auc.append(final_average_precision)
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
                for root, dirs, files in os.walk('./2D/test'):
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
                                abnormal_data, abnormal_label = Read2DDataset(file_name)
                                elem_num = 2
                                if multivariate:
                                    abnormal_data = np.expand_dims(abnormal_data, axis=0)
                                if partition:
                                    splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data, abnormal_label, _part_number=k_partition)
                                    final_error = []
                                    for i in range(k_partition):
                                        error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, pr_cks = RunModel(
                                            splitted_data[i], splitted_label[i])
                                        final_error.append(error_partition)
                                    # print('-----------------------------------------')
                                    final_error = np.concatenate(final_error).ravel()
                                    final_zscore = Z_Score(final_error)
                                    y_pred = CreateLabelBasedOnZscore(final_zscore, 3)
                                    final_p, final_r, final_f = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
                                    PrintPrecisionRecallF1Metrics(final_p, final_r, final_f)
                                    final_fpr, final_tpr, final_average_roc_auc = CalculateROCAUCMetrics(abnormal_label, final_error)
                                    print('roc_auc=' + str(final_average_roc_auc))
                                    final_precision_curve, final_recall_curve, final_average_precision = CalculatePrecisionRecallCurve(
                                        abnormal_label, final_error)
                                    print('pr_auc=' + str(final_average_precision))
                                    cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                                    print('cohen_kappa=' + str(cks))
                                else:
                                    error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label)

                                s_precision.append(final_p)
                                s_recall.append(final_r)
                                s_f1.append(final_f)
                                s_roc_auc.append(final_average_roc_auc)
                                s_pr_auc.append(final_average_precision)
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
                k_partition = 2
                s_precision = []
                s_recall = []
                s_f1 = []
                s_roc_auc = []
                s_pr_auc = []
                s_cks = []
                for root, dirs, _ in os.walk('./UAH/'):
                    for dir in dirs:
                        folder_name = os.path.join(root, dir)
                        print(folder_name)
                        abnormal_data, abnormal_label = ReadUAHDataset(folder_name)
                        elem_num = 4
                        if multivariate:
                            abnormal_data = np.expand_dims(abnormal_data, axis=0)
                        if partition:
                            splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data, abnormal_label,
                                                                                     _part_number=k_partition)
                            final_error = []
                            for i in range(k_partition):
                                error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(
                                    splitted_data[i], splitted_label[i])
                                final_error.append(error_partition)
                            # print('-----------------------------------------')
                            final_error = np.concatenate(final_error).ravel()
                            final_zscore = Z_Score(final_error)
                            y_pred = CreateLabelBasedOnZscore(final_zscore, 3)
                            print('########################################')
                            precision, recall, f1 = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
                            PrintPrecisionRecallF1Metrics(precision, recall, f1)
                            _, _, roc_auc = CalculateROCAUCMetrics(abnormal_label, final_error)
                            print('roc_auc=' + str(roc_auc))
                            _, _, pr_auc = CalculatePrecisionRecallCurve(abnormal_label, final_error)
                            print('pr_auc=' + str(pr_auc))
                            cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                            print('cohen_kappa=' + str(cks))
                            print('########################################')
                        else:
                            error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label)

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
                for root, dirs, files in os.walk('./ECG/'):
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
                                abnormal_data, abnormal_label = ReadECGDataset(file_name)
                                elem_num = 3
                                if multivariate:
                                    abnormal_data = np.expand_dims(abnormal_data, axis=0)
                                if partition:
                                    splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data,
                                                                                             abnormal_label,
                                                                                             _part_number=k_partition)
                                    final_error = []
                                    for i in range(k_partition):
                                        error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, pr_cks = RunModel(
                                            splitted_data[i], splitted_label[i])
                                        final_error.append(error_partition)
                                    # print('-----------------------------------------')
                                    final_error = np.concatenate(final_error).ravel()
                                    final_zscore = Z_Score(final_error)
                                    y_pred = CreateLabelBasedOnZscore(final_zscore, 3)
                                    final_p, final_r, final_f = CalculatePrecisionRecallF1Metrics(abnormal_label,
                                                                                                  y_pred)
                                    PrintPrecisionRecallF1Metrics(final_p, final_r, final_f)
                                    final_fpr, final_tpr, final_average_roc_auc = CalculateROCAUCMetrics(abnormal_label,
                                                                                                         final_error)
                                    print('roc_auc=' + str(final_average_roc_auc))
                                    final_precision_curve, final_recall_curve, final_average_precision = CalculatePrecisionRecallCurve(
                                        abnormal_label, final_error)
                                    print('pr_auc=' + str(final_average_precision))
                                    cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                                    print('cohen_kappa=' + str(cks))
                                else:
                                    error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data,
                                                                                                  abnormal_label)

                                s_precision.append(final_p)
                                s_recall.append(final_r)
                                s_f1.append(final_f)
                                s_roc_auc.append(final_average_roc_auc)
                                s_pr_auc.append(final_average_precision)
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
        # file name parameter
        dataset = int(sys.argv[1])
        if dataset == 1:
            file_name = './GD/data/Genesis_AnomalyLabels.csv'
            print(file_name)
            k_partition = 40
            abnormal_data, abnormal_label = ReadGDDataset(file_name)
            elem_num = 18
            if multivariate:
                abnormal_data = np.expand_dims(abnormal_data, axis=0)
            if partition:
                splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data, abnormal_label, _part_number=k_partition)
                final_error = []
                for i in range(k_partition):
                    error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(splitted_data[i], splitted_label[i])
                    final_error.append(error_partition)
                # print('-----------------------------------------')
                final_error = np.concatenate(final_error).ravel()
                final_zscore = Z_Score(final_error)
                y_pred = CreateLabelBasedOnZscore(final_zscore, 2.5)
                final_p, final_r, final_f = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
                PrintPrecisionRecallF1Metrics(final_p, final_r, final_f)
                final_fpr, final_tpr, final_average_roc_auc = CalculateROCAUCMetrics(abnormal_label, final_error)
                print('roc_auc=' + str(final_average_roc_auc))
                final_precision_curve, final_recall_curve, final_average_precision = CalculatePrecisionRecallCurve(abnormal_label, final_error)
                print('pr_auc=' + str(final_average_precision))
                cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                print('cohen_kappa=' + str(cks))
            else:
                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label)
        
        if dataset == 2:
            file_name = './HSS/data/HRSS_anomalous_standard.csv'
            print(file_name)
            k_partition = 80
            abnormal_data, abnormal_label = ReadHSSDataset(file_name)
            elem_num = 18
            if multivariate:
                abnormal_data = np.expand_dims(abnormal_data, axis=0)
            if partition:
                splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data, abnormal_label, _part_number=k_partition)
                final_error = []
                for i in range(k_partition):
                    error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(splitted_data[i], splitted_label[i])
                    final_error.append(error_partition)
                # print('-----------------------------------------')
                final_error = np.concatenate(final_error).ravel()
                final_zscore = Z_Score(final_error)
                y_pred = CreateLabelBasedOnZscore(final_zscore, 3)
                final_p, final_r, final_f = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
                PrintPrecisionRecallF1Metrics(final_p, final_r, final_f)
                final_fpr, final_tpr, final_average_roc_auc = CalculateROCAUCMetrics(abnormal_label, final_error)
                print('roc_auc=' + str(final_average_roc_auc))
                final_precision_curve, final_recall_curve, final_average_precision = CalculatePrecisionRecallCurve(abnormal_label, final_error)
                print('pr_auc=' + str(final_average_precision))
                cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                print('cohen_kappa=' + str(cks))
            else:
                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label)

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
                            abnormal_data, abnormal_label = ReadS5Dataset(file_name)
                            elem_num = 1
                            if multivariate:
                                abnormal_data = np.expand_dims(abnormal_data, axis=0)
                            if partition:
                                splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data, abnormal_label, _part_number=k_partition)
                                final_error = []
                                for i in range(k_partition):
                                    error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, pr_cks = RunModel(splitted_data[i], splitted_label[i])
                                    final_error.append(error_partition)
                                # print('-----------------------------------------')
                                final_error = np.concatenate(final_error).ravel()
                                final_zscore = Z_Score(final_error)
                                y_pred = CreateLabelBasedOnZscore(final_zscore, 3)
                                final_p, final_r, final_f = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
                                PrintPrecisionRecallF1Metrics(final_p, final_r, final_f)
                                final_fpr, final_tpr, final_average_roc_auc = CalculateROCAUCMetrics(abnormal_label, final_error)
                                print('roc_auc=' + str(final_average_roc_auc))
                                final_precision_curve, final_recall_curve, final_average_precision = CalculatePrecisionRecallCurve(abnormal_label, final_error)
                                print('pr_auc=' + str(final_average_precision))
                                cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                                print('cohen_kappa=' + str(cks))
                            else:
                                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label)

                            s_precision.append(final_p)
                            s_recall.append(final_r)
                            s_f1.append(final_f)
                            s_roc_auc.append(final_average_roc_auc)
                            s_pr_auc.append(final_average_precision)
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
                            abnormal_data, abnormal_label = ReadNABDataset(file_name)
                            elem_num = 1
                            if multivariate:
                                abnormal_data = np.expand_dims(abnormal_data, axis=0)
                            if partition:
                                splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data, abnormal_label, _part_number=k_partition)
                                final_error = []
                                for i in range(k_partition):
                                    error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, pr_cks = RunModel(splitted_data[i], splitted_label[i])
                                    final_error.append(error_partition)
                                # print('-----------------------------------------')
                                final_error = np.concatenate(final_error).ravel()
                                final_zscore = Z_Score(final_error)
                                y_pred = CreateLabelBasedOnZscore(final_zscore, 3)
                                final_p, final_r, final_f = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
                                PrintPrecisionRecallF1Metrics(final_p, final_r, final_f)
                                final_fpr, final_tpr, final_average_roc_auc = CalculateROCAUCMetrics(abnormal_label, final_error)
                                print('roc_auc=' + str(final_average_roc_auc))
                                final_precision_curve, final_recall_curve, final_average_precision = CalculatePrecisionRecallCurve(abnormal_label, final_error)
                                print('pr_auc=' + str(final_average_precision))
                                cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                                print('cohen_kappa=' + str(cks))
                            else:
                                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label)

                            s_precision.append(final_p)
                            s_recall.append(final_r)
                            s_f1.append(final_f)
                            s_roc_auc.append(final_average_roc_auc)
                            s_pr_auc.append(final_average_precision)
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
            for root, dirs, files in os.walk('./2D/test'):
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
                            abnormal_data, abnormal_label = Read2DDataset(file_name)
                            elem_num = 2
                            if multivariate:
                                abnormal_data = np.expand_dims(abnormal_data, axis=0)
                            if partition:
                                splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data, abnormal_label,
                                                                                         _part_number=k_partition)
                                final_error = []
                                for i in range(k_partition):
                                    error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, pr_cks = RunModel(
                                        splitted_data[i], splitted_label[i])
                                    final_error.append(error_partition)
                                # print('-----------------------------------------')
                                final_error = np.concatenate(final_error).ravel()
                                final_zscore = Z_Score(final_error)
                                y_pred = CreateLabelBasedOnZscore(final_zscore, 3)
                                final_p, final_r, final_f = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
                                PrintPrecisionRecallF1Metrics(final_p, final_r, final_f)
                                final_fpr, final_tpr, final_average_roc_auc = CalculateROCAUCMetrics(abnormal_label,
                                                                                                     final_error)
                                print('roc_auc=' + str(final_average_roc_auc))
                                final_precision_curve, final_recall_curve, final_average_precision = CalculatePrecisionRecallCurve(
                                    abnormal_label, final_error)
                                print('pr_auc=' + str(final_average_precision))
                                cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                                print('cohen_kappa=' + str(cks))
                            else:
                                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data,
                                                                                              abnormal_label)

                            s_precision.append(final_p)
                            s_recall.append(final_r)
                            s_f1.append(final_f)
                            s_roc_auc.append(final_average_roc_auc)
                            s_pr_auc.append(final_average_precision)
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
            k_partition = 2
            s_precision = []
            s_recall = []
            s_f1 = []
            s_roc_auc = []
            s_pr_auc = []
            s_cks = []
            for root, dirs, _ in os.walk('./UAH/'):
                for dir in dirs:
                    folder_name = os.path.join(root, dir)
                    print(folder_name)
                    abnormal_data, abnormal_label = ReadUAHDataset(folder_name)
                    elem_num = 4
                    if multivariate:
                        abnormal_data = np.expand_dims(abnormal_data, axis=0)
                    if partition:
                        splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data, abnormal_label,
                                                                                     _part_number=k_partition)
                        final_error = []
                        for i in range(k_partition):
                            error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(
                                splitted_data[i], splitted_label[i])
                            final_error.append(error_partition)
                        # print('-----------------------------------------')
                        final_error = np.concatenate(final_error).ravel()
                        final_zscore = Z_Score(final_error)
                        y_pred = CreateLabelBasedOnZscore(final_zscore, 3)
                        print('########################################')
                        precision, recall, f1 = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
                        PrintPrecisionRecallF1Metrics(precision, recall, f1)
                        _, _, roc_auc = CalculateROCAUCMetrics(abnormal_label, final_error)
                        print('roc_auc=' + str(roc_auc))
                        _, _, pr_auc = CalculatePrecisionRecallCurve(abnormal_label, final_error)
                        print('pr_auc=' + str(pr_auc))
                        cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                        print('cohen_kappa=' + str(cks))
                        print('########################################')
                    else:
                        error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label)

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
            for root, dirs, files in os.walk('./ECG/'):
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
                            abnormal_data, abnormal_label = ReadECGDataset(file_name)
                            elem_num = 3
                            if multivariate:
                                abnormal_data = np.expand_dims(abnormal_data, axis=0)
                            if partition:
                                splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data, abnormal_label,
                                                                                         _part_number=k_partition)
                                final_error = []
                                for i in range(k_partition):
                                    error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, pr_cks = RunModel(
                                        splitted_data[i], splitted_label[i])
                                    final_error.append(error_partition)
                                # print('-----------------------------------------')
                                final_error = np.concatenate(final_error).ravel()
                                final_zscore = Z_Score(final_error)
                                y_pred = CreateLabelBasedOnZscore(final_zscore, 3)
                                final_p, final_r, final_f = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
                                PrintPrecisionRecallF1Metrics(final_p, final_r, final_f)
                                final_fpr, final_tpr, final_average_roc_auc = CalculateROCAUCMetrics(abnormal_label,
                                                                                                     final_error)
                                print('roc_auc=' + str(final_average_roc_auc))
                                final_precision_curve, final_recall_curve, final_average_precision = CalculatePrecisionRecallCurve(
                                    abnormal_label, final_error)
                                print('pr_auc=' + str(final_average_precision))
                                cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                                print('cohen_kappa=' + str(cks))
                            else:
                                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data,
                                                                                              abnormal_label)

                            s_precision.append(final_p)
                            s_recall.append(final_r)
                            s_f1.append(final_f)
                            s_roc_auc.append(final_average_roc_auc)
                            s_pr_auc.append(final_average_precision)
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


