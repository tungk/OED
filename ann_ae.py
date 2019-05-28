import os
import pathlib
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *


def Model(_abnormal_data, _abnormal_label, _hidden_num, _file_name):
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        # placeholder list
        p_input = tf.placeholder(tf.float32, shape=(_abnormal_data.shape[0], _abnormal_data.shape[1]))
        p_input_reshape = tf.reshape(p_input, [batch_num, _abnormal_data.shape[0] * _abnormal_data.shape[1]])

        # initialize weights randomly from a Gaussian distribution
        # step 1: create the initializer for weights
        weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        # step 2: create the weight variable with proper initialization
        w_enc = tf.get_variable(name="weight_enc", dtype=tf.float32, shape=[_abnormal_data.shape[0] * _abnormal_data.shape[1], _hidden_num], initializer=weight_initer)
        w_dec = tf.get_variable(name="weight_dec", dtype=tf.float32, shape=[_hidden_num, _abnormal_data.shape[0] * _abnormal_data.shape[1]], initializer=weight_initer)

        w_enc_sparse = tf.convert_to_tensor(np.random.randint(2, size=(_abnormal_data.shape[0] * _abnormal_data.shape[1], _hidden_num)), dtype=tf.float32)
        w_dec_sparse = tf.convert_to_tensor(np.random.randint(2, size=(_hidden_num, _abnormal_data.shape[0] * _abnormal_data.shape[1])), dtype=tf.float32)

        b_enc = tf.Variable(tf.zeros(_hidden_num), dtype=tf.float32)
        b_dec = tf.Variable(tf.zeros(_abnormal_data.shape[0] * _abnormal_data.shape[1]), dtype=tf.float32)

        bottle_neck = tf.nn.sigmoid(tf.matmul(p_input_reshape, w_enc * w_enc_sparse) + b_enc)
        dec_output = tf.nn.sigmoid(tf.matmul(bottle_neck, w_dec * w_dec_sparse) + b_dec)
        dec_output_reshape = tf.reshape(dec_output, [batch_num, _abnormal_data.shape[0], _abnormal_data.shape[1]])

        loss = tf.reduce_mean(tf.square(p_input - dec_output_reshape))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    return g, p_input, dec_output_reshape, loss, optimizer, saver

def RunModel(_abnormal_data, _abnormal_label, _hidden_num, _file_name):
    error = []
    for j in range(ensemble_space):
        graph, p_input, dec_outputs, loss, optimizer, saver = Model(_abnormal_data, _abnormal_label, _hidden_num, _file_name)
        config = tf.ConfigProto()

        # config.gpu_options.allow_growth = True
        # config.allow_soft_placement=True

        # Add ops to save and restore all the variables.
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

            if save_model:
                save_path = saver.save(sess, './saved_model/' + pathlib.Path(_file_name).parts[
                    0] + '/randnet_' + os.path.basename(_file_name) + '.ckpt')
                print("Model saved in path: %s" % save_path)

            (input_, output_) = sess.run([p_input, dec_outputs], {p_input: _abnormal_data})
            error.append(SquareErrorDataPoints(np.expand_dims(input_, 0), output_))

    ensemble_errors = np.asarray(error)
    anomaly_score = CalculateFinalAnomalyScore(ensemble_errors)

    zscore = Z_Score(anomaly_score)
    y_pred = CreateLabelBasedOnZscore(zscore, 3)
    precision, recall, f1 = CalculatePrecisionRecallF1Metrics(_abnormal_label, y_pred)
    fpr, tpr, roc_auc = CalculateROCAUCMetrics(_abnormal_label, anomaly_score)
    precision_curve, recall_curve, average_precision = CalculatePrecisionRecallCurve(_abnormal_label, anomaly_score)
    cks = CalculateCohenKappaMetrics(_abnormal_label, y_pred)

    return anomaly_score, precision, recall, f1, roc_auc, average_precision, cks

if __name__ == '__main__':
    batch_num = 1
    learning_rate = 1e-3
    hidden_num = 128
    iteration = 50
    ensemble_space = 20
    save_model = False

    try:
        sys.argv[1]
    except IndexError:
        for n in range(1, 6):
            # file name parameter
            dataset = n
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
                                abnormal_data = np.reshape(abnormal_data.shape[0] * abnormal_data.shape[1])
                                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label, hidden_num, _file_name=file_name)
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
                                abnormal_data, abnormal_label = ReadECGDataset(file_name)
                                elem_num = 3
                                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label, hidden_num, _file_name=file_name)

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
        # file name parameter
        dataset = int(sys.argv[1])

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
                            error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label, hidden_num, _file_name=file_name)

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
                    k_partition = 6
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
                            error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label, hidden_num, _file_name=file_name)
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
