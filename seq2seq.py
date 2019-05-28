import pathlib
import sys

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.rnn_cell_impl import _Linear, LSTMStateTuple
from tensorflow.python.ops import variable_scope as vs
from utils import *
from tensorflow.contrib import seq2seq


class RLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    """Basic LSTM recurrent network cell.

    The implementation is based on: http://arxiv.org/abs/1409.2329.

    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.

    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.

    For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
    that follows.
    """

    def __init__(self, num_units, forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None, dense=None):
        """Initialize the basic LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
            Must set to `0.0` manually when restoring from CudnnLSTM-trained
            checkpoints.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.

          When restoring from CudnnLSTM-trained checkpoints, must use
          CudnnCompatibleLSTMCell instead.
        """
        super(tf.nn.rnn_cell.BasicLSTMCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            tf.logging.warn("%s: Using a concatenated state is slower and will soon be "
                            "deprecated.  Use state_is_tuple=True.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or tf.nn.tanh
        self._linear = None
        self._dense = dense

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).

        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size x 2 * self.state_size]`.

        Returns:
          A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        sigmoid = tf.sigmoid

        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

        if self._linear is None:
            self._linear = _Linear([inputs, h], 4 * self._num_units, True)

        i, j, f, o = tf.split(value=self._linear([inputs, h]), num_or_size_splits=4, axis=1)

        new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        new_h_1 = self._activation(new_c) * sigmoid(o)

        w_h, b_h = self.weight_bias([self._num_units, self._num_units], [self._num_units])
        new_h_2 = sigmoid(tf.matmul(h, w_h) + b_h)
        masked_w1, masked_w2 = self.masked_weight()
        new_h = new_h_1 * masked_w1 + new_h_2 * masked_w2

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = tf.concat([new_c, new_h], 1)
        return new_h, new_state

    def weight_bias(self, W_shape, b_shape, bias_init=0.1):
        """Fully connected highway layer adopted from
           https://github.com/fomorians/highway-fcn/blob/master/main.py
        """
        W = tf.get_variable("weight_h_2", shape=W_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable("bias_h_2", shape=b_shape)
        return W, b

    def masked_weight(self):
        masked_W1 = np.random.randint(2, size=1)
        if masked_W1 == 0:
            masked_W2 = 1
        else:
            masked_W2 = np.random.randint(2, size=1)
        tf_mask_W1 = tf.constant(masked_W1, dtype=tf.float32)
        tf_mask_W2 = tf.constant(masked_W2, dtype=tf.float32)
        return tf_mask_W1, tf_mask_W2


class RGRUCell(tf.nn.rnn_cell.GRUCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self, num_units, activation=None, reuse=None, kernel_initializer=None, bias_initializer=None):
        super(tf.nn.rnn_cell.GRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or tf.nn.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""

        sigmoid = tf.sigmoid

        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                self._gate_linear = _Linear([inputs, state], 2 * self._num_units, True, bias_initializer=bias_ones, kernel_initializer=self._kernel_initializer)

        value = sigmoid(self._gate_linear([inputs, state]))
        r, u = tf.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear([inputs, r_state], self._num_units, True, bias_initializer=self._bias_initializer, kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        new_h = (u * state + (1 - u) * c)
        return new_h, new_h


    def weight_bias(self, W_shape, b_shape, bias_init=0.1):
        """Fully connected highway layer adopted from
           https://github.com/fomorians/highway-fcn/blob/master/main.py
        """
        W = tf.get_variable("weight_h_2", shape=W_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable("bias_h_2", shape=b_shape)
        return W, b


def Model(_abnormal_data, _abnormal_label, _hidden_num, _elem_num):
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        # placeholder list
        p_input = tf.placeholder(tf.float32, shape=(batch_num, _abnormal_data.shape[1], _abnormal_data.shape[2]))

        # create RNN cell
        if cell_type == 0:
            enc_cell = tf.nn.rnn_cell.BasicRNNCell(_hidden_num)
            dec_cell = tf.nn.rnn_cell.BasicRNNCell(_hidden_num)
        if cell_type == 1:
            enc_cell = tf.nn.rnn_cell.LSTMCell(_hidden_num)
            dec_cell = tf.nn.rnn_cell.LSTMCell(_hidden_num)
        if cell_type == 2:
            enc_cell = tf.nn.rnn_cell.GRUCell(_hidden_num)
            dec_cell = tf.nn.rnn_cell.GRUCell(_hidden_num)

        projection_layer = tf.layers.Dense(units=_elem_num, use_bias=True)

        # with tf.device('/device:GPU:0'):
        with tf.variable_scope('encoder'):
            (enc_output, enc_state) = tf.nn.dynamic_rnn(enc_cell, p_input, dtype=tf.float32)

        # with tf.device('/device:GPU:1'):
        with tf.variable_scope('decoder') as vs:
            # dec_weight_ = tf.Variable(tf.truncated_normal([hidden_num, elem_num], dtype=tf.float32), name='dec_weight')
            # dec_bias_ = tf.Variable(tf.constant(0.1, shape=[elem_num], dtype=tf.float32), name='dec_bias')
            if decode_without_input:
                dec_input = tf.zeros([p_input.shape[0], p_input.shape[1], p_input.shape[2]], dtype=tf.float32)
                (dec_outputs, dec_state_) = tf.nn.dynamic_rnn(dec_cell, dec_input, initial_state=enc_state, dtype=tf.float32)
                if reverse:
                    dec_outputs = dec_outputs[::-1]
            else:
                dec_input = tf.zeros([p_input.shape[0], p_input.shape[2]], dtype=tf.float32)
                inference_helper = tf.contrib.seq2seq.InferenceHelper(
                    sample_fn=lambda outputs: outputs,
                    sample_shape=[_elem_num],
                    sample_dtype=tf.float32,
                    start_inputs=dec_input,
                    end_fn=lambda sample_ids: False)
                inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                                    inference_helper,
                                                                    enc_state,
                                                                    output_layer=projection_layer)
                dec_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    inference_decoder, impute_finished=True,
                    maximum_iterations=p_input.shape[1])

                if reverse:
                    dec_outputs = dec_outputs[::-1]

        loss = tf.reduce_mean(tf.square(p_input - dec_outputs))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
    return g, p_input, dec_outputs, loss, optimizer, saver


def RunModel(_abnormal_data, _abnormal_label, _hidden_num, _elem_num):
    graph, p_input, dec_outputs, loss, optimizer, saver = Model(_abnormal_data, _abnormal_label, _hidden_num, _elem_num)
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
            save_path = saver.save(sess, './saved_model/' + pathlib.Path(file_name).parts[0] + '/rnn_seq2seq_' + str(cell_type) + '_' + os.path.basename(file_name) + '.ckpt')
            print("Model saved in path: %s" % save_path)

        (input_, output_) = sess.run([p_input, dec_outputs], {p_input: _abnormal_data})
        sess.close()
    del sess
    error = SquareErrorDataPoints(input_, output_[0])

    # np.savetxt('./saved_result/' + pathlib.Path(file_name).parts[0] + '/rnn_seq2seq_' + os.path.basename(file_name) + '_error.txt', error, delimiter=',')  # X is an array
    zscore = Z_Score(error)
    # np.savetxt('./saved_result/' + pathlib.Path(file_name).parts[0] + '/rnn_seq2seq_' + os.path.basename(file_name) + '_zscore.txt', zscore, delimiter=',')  # X is an array

    y_pred = CreateLabelBasedOnZscore(zscore, 3)

    if not partition:
        score_pred_label = np.c_[error, y_pred, _abnormal_label]
        np.savetxt('./saved_result/' + pathlib.Path(file_name).parts[0] + '/rnn_seq2seq_' + os.path.basename(file_name) + '_score.txt', score_pred_label, delimiter=',')  # X is an array

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
    hidden_num = 16
    # step_num = 8
    iteration = 50
    ensemble_space = 20
    learning_rate = 1e-3
    multivariate = True

    reverse = True
    decode_without_input = False

    partition = True
    save_model = False

    # cell type 0 => BasicRNN, 1 => LSTM, 2 => GRU
    cell_type = 1
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
                        error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(splitted_data[i], splitted_label[i], hidden_num, elem_num)
                        final_error.append(error_partition)
                    # print('-----------------------------------------')
                    final_error = np.concatenate(final_error).ravel()
                    final_zscore = Z_Score(final_error)
                    y_pred = CreateLabelBasedOnZscore(final_zscore, 3)
                    precision, recall, f1 = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
                    PrintPrecisionRecallF1Metrics(precision, recall, f1)
                    _, _, roc_auc = CalculateROCAUCMetrics(abnormal_label, final_error)
                    print('roc_auc=' + str(roc_auc))
                    _, _, pr_auc = CalculatePrecisionRecallCurve(abnormal_label, final_error)
                    print('pr_auc=' + str(pr_auc))
                    cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                    print('cohen_kappa=' + str(cks))
                else:
                    error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label, hidden_num, elem_num)
            
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
                        error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(splitted_data[i], splitted_label[i], hidden_num, elem_num)
                        final_error.append(error_partition)
                    # print('-----------------------------------------')
                    final_error = np.concatenate(final_error).ravel()
                    final_zscore = Z_Score(final_error)
                    y_pred = CreateLabelBasedOnZscore(final_zscore, 3)
                    precision, recall, f1 = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
                    PrintPrecisionRecallF1Metrics(precision, recall, f1)
                    _, _, roc_auc = CalculateROCAUCMetrics(abnormal_label, final_error)
                    print('roc_auc=' + str(roc_auc))
                    _, _, pr_auc = CalculatePrecisionRecallCurve(abnormal_label, final_error)
                    print('pr_auc=' + str(pr_auc))
                    cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                    print('cohen_kappa=' + str(cks))
                else:
                    error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label, hidden_num, elem_num)

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
                                        error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(splitted_data[i], splitted_label[i], hidden_num, elem_num)
                                        final_error.append(error_partition)
                                    # print('-----------------------------------------')
                                    final_error = np.concatenate(final_error).ravel()
                                    final_zscore = Z_Score(final_error)
                                    y_pred = CreateLabelBasedOnZscore(final_zscore, 3)
                                    precision, recall, f1 = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
                                    PrintPrecisionRecallF1Metrics(precision, recall, f1)
                                    _, _, roc_auc = CalculateROCAUCMetrics(abnormal_label, final_error)
                                    print('roc_auc=' + str(roc_auc))
                                    _, _, pr_auc = CalculatePrecisionRecallCurve(abnormal_label, final_error)
                                    print('pr_auc=' + str(pr_auc))
                                    cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                                    print('cohen_kappa=' + str(cks))
                                else:
                                    error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label, hidden_num, elem_num)

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
                                abnormal_data, abnormal_label = ReadNABDataset(file_name)
                                elem_num = 1
                                if multivariate:
                                    abnormal_data = np.expand_dims(abnormal_data, axis=0)
                                if partition:
                                    splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data, abnormal_label,
                                                                                             _part_number=k_partition)
                                    final_error = []
                                    for i in range(k_partition):
                                        error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(splitted_data[i], splitted_label[i], hidden_num, elem_num)
                                        final_error.append(error_partition)
                                    # print('-----------------------------------------')
                                    final_error = np.concatenate(final_error).ravel()
                                    final_zscore = Z_Score(final_error)
                                    y_pred = CreateLabelBasedOnZscore(final_zscore, 3)
                                    precision, recall, f1 = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
                                    PrintPrecisionRecallF1Metrics(precision, recall, f1)
                                    _, _, roc_auc = CalculateROCAUCMetrics(abnormal_label, final_error)
                                    print('roc_auc=' + str(roc_auc))
                                    _, _, pr_auc = CalculatePrecisionRecallCurve(abnormal_label, final_error)
                                    print('pr_auc=' + str(pr_auc))
                                    cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                                    print('cohen_kappa=' + str(cks))
                                else:
                                    error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label, hidden_num, elem_num)

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
                                    splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data,
                                                                                             abnormal_label,
                                                                                             _part_number=k_partition)
                                    final_error = []
                                    for i in range(k_partition):
                                        error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, pr_cks = RunModel(
                                            splitted_data[i], splitted_label[i], hidden_num, elem_num)
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
                                                                                                  abnormal_label, hidden_num, elem_num)

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
                                    splitted_data[i], splitted_label[i], hidden_num, elem_num)
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
                            error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label,
                                                                                          hidden_num, elem_num)

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
                                            splitted_data[i], splitted_label[i], hidden_num, elem_num)
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
                                                                                                  abnormal_label,
                                                                                                  hidden_num, elem_num)

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
                    error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(splitted_data[i], splitted_label[i], hidden_num, elem_num)
                    final_error.append(error_partition)
                # print('-----------------------------------------')
                final_error = np.concatenate(final_error).ravel()
                final_zscore = Z_Score(final_error)
                y_pred = CreateLabelBasedOnZscore(final_zscore, 3)
                precision, recall, f1 = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
                PrintPrecisionRecallF1Metrics(precision, recall, f1)
                _, _, roc_auc = CalculateROCAUCMetrics(abnormal_label, final_error)
                print('roc_auc=' + str(roc_auc))
                _, _, pr_auc = CalculatePrecisionRecallCurve(abnormal_label, final_error)
                print('pr_auc=' + str(pr_auc))
                cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                print('cohen_kappa=' + str(cks))
            else:
                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label, hidden_num, elem_num)
        
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
                    error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(splitted_data[i], splitted_label[i], hidden_num, elem_num)
                    final_error.append(error_partition)
                # print('-----------------------------------------')
                final_error = np.concatenate(final_error).ravel()
                final_zscore = Z_Score(final_error)
                y_pred = CreateLabelBasedOnZscore(final_zscore, 3)
                precision, recall, f1 = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
                PrintPrecisionRecallF1Metrics(precision, recall, f1)
                _, _, roc_auc = CalculateROCAUCMetrics(abnormal_label, final_error)
                print('roc_auc=' + str(roc_auc))
                _, _, pr_auc = CalculatePrecisionRecallCurve(abnormal_label, final_error)
                print('pr_auc=' + str(pr_auc))
                cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                print('cohen_kappa=' + str(cks))
            else:
                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label, hidden_num, elem_num)

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
                                splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data, abnormal_label,
                                                                                         _part_number=k_partition)
                                final_error = []
                                for i in range(k_partition):
                                    error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(
                                        splitted_data[i], splitted_label[i], hidden_num, elem_num)
                                    final_error.append(error_partition)
                                # print('-----------------------------------------')
                                final_error = np.concatenate(final_error).ravel()
                                final_zscore = Z_Score(final_error)
                                y_pred = CreateLabelBasedOnZscore(final_zscore, 3)
                                precision, recall, f1 = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
                                PrintPrecisionRecallF1Metrics(precision, recall, f1)
                                _, _, roc_auc = CalculateROCAUCMetrics(abnormal_label, final_error)
                                print('roc_auc=' + str(roc_auc))
                                _, _, pr_auc = CalculatePrecisionRecallCurve(abnormal_label, final_error)
                                print('pr_auc=' + str(pr_auc))
                                cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                                print('cohen_kappa=' + str(cks))
                            else:
                                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label,
                                                                                              hidden_num, elem_num)

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
                            abnormal_data, abnormal_label = ReadNABDataset(file_name)
                            elem_num = 1
                            if multivariate:
                                abnormal_data = np.expand_dims(abnormal_data, axis=0)
                            if partition:
                                splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data, abnormal_label,
                                                                                         _part_number=k_partition)
                                final_error = []
                                for i in range(k_partition):
                                    error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(
                                        splitted_data[i], splitted_label[i], hidden_num, elem_num)
                                    final_error.append(error_partition)
                                # print('-----------------------------------------')
                                final_error = np.concatenate(final_error).ravel()
                                final_zscore = Z_Score(final_error)
                                y_pred = CreateLabelBasedOnZscore(final_zscore, 3)
                                precision, recall, f1 = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
                                PrintPrecisionRecallF1Metrics(precision, recall, f1)
                                _, _, roc_auc = CalculateROCAUCMetrics(abnormal_label, final_error)
                                print('roc_auc=' + str(roc_auc))
                                _, _, pr_auc = CalculatePrecisionRecallCurve(abnormal_label, final_error)
                                print('pr_auc=' + str(pr_auc))
                                cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
                                print('cohen_kappa=' + str(cks))
                            else:
                                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label,
                                                                                              hidden_num, elem_num)

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
                                splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data,
                                                                                         abnormal_label,
                                                                                         _part_number=k_partition)
                                final_error = []
                                for i in range(k_partition):
                                    error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, pr_cks = RunModel(
                                        splitted_data[i], splitted_label[i], hidden_num, elem_num)
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
                                                                                              abnormal_label,
                                                                                              hidden_num, elem_num)

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
                                splitted_data[i], splitted_label[i], hidden_num, elem_num)
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
                        error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label,
                                                                                      hidden_num, elem_num)

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
                                        splitted_data[i], splitted_label[i], hidden_num, elem_num)
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
                                                                                              abnormal_label,
                                                                                              hidden_num, elem_num)

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