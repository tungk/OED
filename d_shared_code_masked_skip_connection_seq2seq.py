import pathlib
import copy
import sys

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.rnn_cell_impl import _Linear, LSTMStateTuple, GRUCell, LSTMCell
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

        new_h = new_h_1 + new_h_2

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


class RSLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    """Basic LSTM recurrent network cell.

    The implementation is based on: http://arxiv.org/abs/1409.2329.

    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.

    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.

    For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
    that follows.
    """

    def __init__(self, num_units, forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None, dense=None,
                 file_name='tweet', type='enc', component=1, partition=1):
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
        self._step = 0
        self._file_name = file_name
        self._type = type
        self._component = component
        self._partition = partition

        if not os.path.exists('./weight/' + self._file_name):
            os.makedirs('./weight/' + self._file_name)
        if not os.path.exists('./weight/' + self._file_name + '/' + str(self._partition)):
            os.makedirs('./weight/' + self._file_name + '/' + str(self._partition))
        if not os.path.exists('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component)):
            os.makedirs('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component))
        if not os.path.exists('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(
                self._component) + '/' + self._type):
            os.makedirs('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(
                self._component) + '/' + self._type)

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
        self._step = self._step + 1

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
        masked_w1, masked_w2 = self.masked_weight(_load=False)

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

    def masked_weight(self, _load=False):
        if _load == False:
            masked_W1 = np.random.randint(2, size=1)
            if masked_W1 == 0:
                masked_W2 = 1
            else:
                masked_W2 = np.random.randint(2, size=1)

            np.save('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component) + '/'
                    + self._type + '/W1_' + str(self._step), masked_W1)
            np.save('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component) + '/'
                    + self._type + '/W2_' + str(self._step), masked_W2)
        else:
            masked_W1 = np.load('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component)
                                + '/' + str(self._type) + '/W1_' + str(self._step) + '.npy')
            masked_W2 = np.load('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component)
                                + '/' + str(self._type) + '/W2_' + str(self._step) + '.npy')

        tf_mask_W1 = tf.constant(masked_W1, dtype=tf.float32)
        tf_mask_W2 = tf.constant(masked_W2, dtype=tf.float32)
        return tf_mask_W1, tf_mask_W2


class RKLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    """Basic LSTM recurrent network cell.

    The implementation is based on: http://arxiv.org/abs/1409.2329.

    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.

    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.

    For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
    that follows.
    """

    def __init__(self, num_units, forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None, dense=None,
                 file_name='tweet', type='enc', component=1, partition=1, skip_size=5):
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
        self._step = 0
        self._file_name = file_name
        self._type = type
        self._component = component
        self._partition = partition
        self._skip_size = skip_size

        if not os.path.exists('./weight/' + self._file_name):
            os.makedirs('./weight/' + self._file_name)
        if not os.path.exists('./weight/' + self._file_name + '/' + str(self._partition)):
            os.makedirs('./weight/' + self._file_name + '/' + str(self._partition))
        if not os.path.exists('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component)):
            os.makedirs('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component))
        if not os.path.exists('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component) + '/' + self._type):
            os.makedirs('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component) + '/' + self._type)

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
        self._step = self._step + 1

        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

        if self._linear is None:
            self._linear = _Linear([inputs, h], 4 * self._num_units, True)

        i, j, f, o = tf.split(value=self._linear([inputs, h]), num_or_size_splits=4, axis=1)

        new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        new_h_cnt = self._activation(new_c) * sigmoid(o)

        if self._step % self._skip_size == 0:
            w_h_skip, b_h_skip = self.weight_bias([self._num_units, self._num_units], [self._num_units])
            new_h_skip = sigmoid(tf.matmul(h, w_h_skip) + b_h_skip)
            masked_w1, masked_w2 = self.masked_weight(_load=False)
            new_h = new_h_cnt * masked_w1 + new_h_skip * masked_w2

        else:
            new_h = new_h_cnt

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


    def masked_weight(self, _load=False):
        if _load==False:
            masked_W1 = np.random.randint(2, size=1)
            if masked_W1 == 0:
                masked_W2 = 1
            else:
                masked_W2 = np.random.randint(2, size=1)

            np.save('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component) + '/'
                    + self._type + '/W1_' + str(self._step), masked_W1)
            np.save('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component) + '/'
                    + self._type + '/W2_' + str(self._step), masked_W2)
        else:
            masked_W1 = np.load('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component)
                                + '/' + str(self._type) + '/W1_' + str(self._step) + '.npy')
            masked_W2 = np.load('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component)
                                + '/' + str(self._type) + '/W2_' + str(self._step) + '.npy')

        tf_mask_W1 = tf.constant(masked_W1, dtype=tf.float32)
        tf_mask_W2 = tf.constant(masked_W2, dtype=tf.float32)
        return tf_mask_W1, tf_mask_W2


class RSGRUCell(tf.nn.rnn_cell.GRUCell):
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
                self._gate_linear = _Linear([inputs, state], 2 * self._num_units, True, bias_initializer=bias_ones,
                                            kernel_initializer=self._kernel_initializer)

        value = sigmoid(self._gate_linear([inputs, state]))
        r, u = tf.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear([inputs, r_state], self._num_units, True,
                                                 bias_initializer=self._bias_initializer,
                                                 kernel_initializer=self._kernel_initializer)
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

def Model(_abnormal_data, _abnormal_label, _hidden_num, _elem_num, _file_name, _partition):
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        # placeholder list
        p_input = tf.placeholder(tf.float32, shape=(batch_num, _abnormal_data.shape[1], _abnormal_data.shape[2]))
        # p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, _abnormal_data.shape[1], 1)]

        # Regularizer signature
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)

        # Projection layer
        projection_layer = tf.layers.Dense(units=_elem_num, use_bias=True)

        # with tf.device('/device:GPU:0'):
        d_enc = {}
        with tf.variable_scope('encoder'):
            for j in range(ensemble_space):
                # create RNN cell
                if cell_type == 0:
                    enc_cell = tf.nn.rnn_cell.BasicRNNCell(_hidden_num)
                if cell_type == 1:
                    pure_enc_cell = LSTMCell(_hidden_num)
                    residual_enc_cell = RLSTMCell(_hidden_num)
                    # enc_cell = RSLSTMCell(_hidden_num, file_name=_file_name, type='enc', partition=_partition,
                    #                       component=j, reuse=tf.AUTO_REUSE)
                    enc_cell = RKLSTMCell(_hidden_num, file_name=_file_name, type='enc', partition=_partition,
                                          component=j, reuse=tf.AUTO_REUSE)
                if cell_type == 2:
                    pure_enc_cell = GRUCell(_hidden_num)
                    enc_cell = RSGRUCell(_hidden_num)
                if j == 0:
                    d_enc['enc_output_{0}'.format(j)], d_enc['enc_state_{0}'.format(j)] = tf.nn.dynamic_rnn(
                        pure_enc_cell, p_input, dtype=tf.float32)

                elif j == 1:
                    d_enc['enc_output_{0}'.format(j)], d_enc['enc_state_{0}'.format(j)] = tf.nn.dynamic_rnn(
                        residual_enc_cell, p_input, dtype=tf.float32)

                else:
                    d_enc['enc_output_{0}'.format(j)], d_enc['enc_state_{0}'.format(j)] = tf.nn.dynamic_rnn(enc_cell,
                                                                                                            p_input,
                                                                                                            dtype=tf.float32)

            # shared_state_c = tf.concat([d_enc['enc_state_{0}'.format(j)].c for j in range(ensemble_space)], axis=1)
            # shared_state_h = tf.concat([d_enc['enc_state_{0}'.format(j)].h for j in range(ensemble_space)], axis=1)
            w_c = tf.Variable(tf.zeros([_hidden_num, _hidden_num]))
            b_c = tf.Variable(tf.zeros([_hidden_num]))
            w_h = tf.Variable(tf.zeros([_hidden_num, _hidden_num]))
            b_h = tf.Variable(tf.zeros([_hidden_num]))
            shared_state_c = tf.concat([tf.matmul(d_enc['enc_state_{0}'.format(j)].c, w_c) + b_c for j in range(ensemble_space)], axis=1)
            shared_state_h = tf.concat([tf.matmul(d_enc['enc_state_{0}'.format(j)].h, w_h) + b_h for j in range(ensemble_space)], axis=1)

            if compress:
                compress_state = tf.layers.Dense(units=_hidden_num, activation=tf.tanh, use_bias=True)
                shared_state_c = compress_state(shared_state_c)
                shared_state_h = compress_state(shared_state_h)

            shared_state = LSTMStateTuple(shared_state_c, shared_state_h)

        # with tf.device('/device:GPU:1'):
        d_dec = {}
        with tf.variable_scope('decoder') as vs:
            if decode_without_input:
                dec_input = tf.zeros([p_input.shape[0], p_input.shape[1], p_input.shape[2]], dtype=tf.float32)
                for k in range(ensemble_space):
                    # create RNN cell
                    if cell_type == 0:
                        dec_cell = tf.nn.rnn_cell.BasicRNNCell(_hidden_num)
                    if cell_type == 1:
                        if compress:
                            pure_dec_cell = LSTMCell(_hidden_num)
                            residual_dec_cell = RLSTMCell(_hidden_num)
                            dec_cell = RSLSTMCell(_hidden_num, file_name=_file_name, type='dec', partition=_partition,
                                                  component=k, reuse=tf.AUTO_REUSE)
                        else:
                            pure_dec_cell = LSTMCell(_hidden_num * ensemble_space)
                            residual_dec_cell = RLSTMCell(_hidden_num * ensemble_space)
                            dec_cell = RSLSTMCell(_hidden_num * ensemble_space, file_name=_file_name, type='dec',
                                                  partition=_partition, component=k, reuse=tf.AUTO_REUSE)
                    if cell_type == 2:
                        if compress:
                            pure_dec_cell = GRUCell(_hidden_num)
                            dec_cell = RSGRUCell(_hidden_num)
                        else:
                            pure_dec_cell = GRUCell(_hidden_num * ensemble_space)
                            dec_cell = RSGRUCell(_hidden_num * ensemble_space)

                    if k == 0:
                        d_dec['dec_output_{0}'.format(k)], d_dec['dec_state_{0}'.format(k)] = tf.nn.dynamic_rnn(
                            pure_dec_cell, dec_input, initial_state=shared_state, dtype=tf.float32)
                    elif k == 1:
                        d_dec['dec_output_{0}'.format(k)], d_dec['dec_state_{0}'.format(k)] = tf.nn.dynamic_rnn(
                            residual_dec_cell, dec_input, initial_state=shared_state, dtype=tf.float32)
                    else:
                        d_dec['dec_output_{0}'.format(k)], d_dec['dec_state_{0}'.format(k)] = tf.nn.dynamic_rnn(
                            dec_cell, dec_input, initial_state=shared_state, dtype=tf.float32)

                    if reverse:
                        d_dec['dec_output_{0}'.format(k)] = d_dec['dec_output_{0}'.format(k)][::-1]

            else:
                dec_input = tf.zeros([p_input.shape[0], p_input.shape[2]], dtype=tf.float32)
                for k in range(ensemble_space):
                    # create RNN cell
                    if cell_type == 0:
                        dec_cell = tf.nn.rnn_cell.BasicRNNCell(_hidden_num)
                    if cell_type == 1:
                        if compress:
                            pure_dec_cell = LSTMCell(_hidden_num)
                            residual_dec_cell = RLSTMCell(_hidden_num)
                            # dec_cell = RSLSTMCell(_hidden_num, file_name=_file_name, type='dec', partition=_partition,
                            #                       component=k, reuse=tf.AUTO_REUSE)
                            dec_cell = RKLSTMCell(_hidden_num, file_name=_file_name, type='dec', partition=_partition,
                                                  component=k, reuse=tf.AUTO_REUSE)
                        else:
                            pure_dec_cell = LSTMCell(_hidden_num * ensemble_space)
                            residual_dec_cell = RLSTMCell(_hidden_num * ensemble_space)
                            # dec_cell = RSLSTMCell(_hidden_num * ensemble_space, file_name=_file_name, type='dec',
                            #                       partition=_partition, component=k, reuse=tf.AUTO_REUSE)
                            dec_cell = RKLSTMCell(_hidden_num * ensemble_space, file_name=_file_name, type='dec',
                        partition = _partition, component=k, reuse=tf.AUTO_REUSE)
                    if cell_type == 2:
                        if compress:
                            pure_dec_cell = GRUCell(_hidden_num)
                            dec_cell = RSGRUCell(_hidden_num)
                        else:
                            pure_dec_cell = GRUCell(_hidden_num * ensemble_space)
                            dec_cell = RSGRUCell(_hidden_num * ensemble_space)

                    inference_helper = tf.contrib.seq2seq.InferenceHelper(
                        sample_fn=lambda outputs: outputs,
                        sample_shape=[_elem_num],
                        sample_dtype=tf.float32,
                        start_inputs=dec_input,
                        end_fn=lambda sample_ids: False)
                    if k == 0:
                        inference_decoder = tf.contrib.seq2seq.BasicDecoder(pure_dec_cell, inference_helper,
                                                                            shared_state, output_layer=projection_layer)
                    elif k == 1:
                        inference_decoder = tf.contrib.seq2seq.BasicDecoder(residual_dec_cell, inference_helper,
                                                                            shared_state, output_layer=projection_layer)
                    else:
                        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, inference_helper, shared_state,
                                                                            output_layer=projection_layer)

                    d_dec['dec_output_{0}'.format(k)], _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                                                impute_finished=True,
                                                                                                maximum_iterations=
                                                                                                p_input.shape[1])

                    if reverse:
                        d_dec['dec_output_{0}'.format(k)] = d_dec['dec_output_{0}'.format(k)][::-1]

        sum_of_difference = 0
        for i in range(ensemble_space):
            sum_of_difference += d_dec['dec_output_{0}'.format(i)][0] - p_input

        loss = tf.reduce_mean(tf.square(sum_of_difference))
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, [shared_state])
        loss = loss + regularization_penalty
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
    return g, p_input, d_dec, loss, optimizer, saver


def RunModel(_abnormal_data, _abnormal_label, _hidden_num, _elem_num, _file_name, _partition):
    graph, p_input, d_dec, loss, optimizer, saver = Model(_abnormal_data, _abnormal_label, _hidden_num, _elem_num,
                                                          _file_name, _partition)
    config = tf.ConfigProto()

    # config.gpu_options.allow_growth = True

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
            save_path = saver.save(sess, './saved_model/' + pathlib.Path(file_name).parts[
                0] + '/shared_code_masked_skip_rnn_seq2seq_' + str(cell_type) + '_' + os.path.basename(
                file_name) + '.ckpt')
            print("Model saved in path: %s" % save_path)

        result = {}
        error = []
        for k in range(ensemble_space):
            (result['input_{0}'.format(k)], result['output_{0}'.format(k)]) = sess.run(
                [p_input, d_dec['dec_output_{0}'.format(k)]], {p_input: _abnormal_data})
            error.append(SquareErrorDataPoints(result['input_{0}'.format(k)], result['output_{0}'.format(k)][0]))

        sess.close()

    ensemble_errors = np.asarray(error)
    anomaly_score = CalculateFinalAnomalyScore(ensemble_errors)
    zscore = Z_Score(anomaly_score)
    y_pred = CreateLabelBasedOnZscore(zscore, 3)
    if not partition:
        score_pred_label = np.c_[ensemble_errors, y_pred, _abnormal_label]
        np.savetxt('./saved_result/' + pathlib.Path(file_name).parts[
            0] + '/shared_code_masked_skip_rnn_seq2seq_' + os.path.basename(file_name) + '_score.txt', score_pred_label,
                   delimiter=',')  # X is an array

    precision, recall, f1 = CalculatePrecisionRecallF1Metrics(_abnormal_label, y_pred)
    if not partition:
        PrintPrecisionRecallF1Metrics(precision, recall, f1)

    fpr, tpr, roc_auc = CalculateROCAUCMetrics(_abnormal_label, anomaly_score)
    # PlotROCAUC(fpr, tpr, roc_auc)
    if not partition:
        print('roc_auc=' + str(roc_auc))

    precision_curve, recall_curve, average_precision = CalculatePrecisionRecallCurve(_abnormal_label, anomaly_score)
    # PlotPrecisionRecallCurve(precision_curve, recall_curve, average_precision)
    if not partition:
        print('pr_auc=' + str(average_precision))

    cks = CalculateCohenKappaMetrics(_abnormal_label, y_pred)
    if not partition:
        print('cks=' + str(cks))

    return anomaly_score, precision, recall, f1, roc_auc, average_precision, cks


if __name__ == '__main__':
    batch_num = 1
    hidden_num = 16
    # step_num = 8
    iteration = 30
    learning_rate = 1e-3
    multivariate = True

    reverse = True
    decode_without_input = False
    compress = False
    partition = True
    save_model = False

    # cell type 0 => BasicRNN, 1 => LSTM, 2 => GRU
    cell_type = 1
    try:
        sys.argv[1]
        sys.argv[2]
    except IndexError:
        ensemble_space = 20
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
                                    splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data,
                                                                                             abnormal_label,
                                                                                             _part_number=k_partition)
                                    final_error = []
                                    for i in range(k_partition):
                                        error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(
                                            splitted_data[i], splitted_label[i], hidden_num, elem_num,
                                            _file_name=os.path.splitext(os.path.basename(file_name))[0], _partition=i)
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
                                    error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data,
                                                                                                  abnormal_label,
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
                                    splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data,
                                                                                             abnormal_label,
                                                                                             _part_number=k_partition)
                                    final_error = []
                                    for i in range(k_partition):
                                        error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(
                                            splitted_data[i], splitted_label[i], hidden_num, elem_num,
                                            _file_name=os.path.splitext(os.path.basename(file_name))[0], _partition=i)
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
                                    error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data,
                                                                                                  abnormal_label,
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
                                        error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(
                                            splitted_data[i], splitted_label[i], hidden_num, elem_num,
                                            _file_name=os.path.splitext(os.path.basename(file_name))[0], _partition=i)
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
                                    error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data,
                                                                                                  abnormal_label,
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

            if dataset == 6:
                k_partition = 3
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
                                    splitted_data[i], splitted_label[i], hidden_num, elem_num, _file_name=dir,
                                    _partition=i)
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
                                if multivariate:
                                    abnormal_data = np.expand_dims(abnormal_data, axis=0)
                                if partition:
                                    splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data,
                                                                                             abnormal_label,
                                                                                             _part_number=k_partition)
                                    final_error = []
                                    for i in range(k_partition):
                                        error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(
                                            splitted_data[i], splitted_label[i], hidden_num, elem_num,
                                            _file_name=os.path.splitext(os.path.basename(file_name))[0], _partition=i)
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
                                    error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data,
                                                                                                  abnormal_label,
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
    else:
        # file name parameter
        dataset = int(sys.argv[1])
        ensemble_space = int(sys.argv[2])
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
                                        splitted_data[i], splitted_label[i], hidden_num, elem_num,
                                        _file_name=os.path.splitext(os.path.basename(file_name))[0], _partition=i)
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
                                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data,
                                                                                              abnormal_label,
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
                                        splitted_data[i], splitted_label[i], hidden_num, elem_num,
                                        _file_name=os.path.splitext(os.path.basename(file_name))[0], _partition=i)
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
                                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data,
                                                                                              abnormal_label,
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
                                    error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(
                                        splitted_data[i], splitted_label[i], hidden_num, elem_num,
                                        _file_name=os.path.splitext(os.path.basename(file_name))[0], _partition=i)
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
                                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data,
                                                                                              abnormal_label,
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

        if dataset == 6:
            k_partition = 3
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
                                splitted_data[i], splitted_label[i], hidden_num, elem_num, _file_name=dir, _partition=i)
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
                            if multivariate:
                                abnormal_data = np.expand_dims(abnormal_data, axis=0)
                            if partition:
                                splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data,
                                                                                         abnormal_label,
                                                                                         _part_number=k_partition)
                                final_error = []
                                for i in range(k_partition):
                                    error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(
                                        splitted_data[i], splitted_label[i], hidden_num, elem_num,
                                        _file_name=os.path.splitext(os.path.basename(file_name))[0], _partition=i)
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
                                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data,
                                                                                              abnormal_label,
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
