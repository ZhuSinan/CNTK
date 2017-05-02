import pytest
import numpy as np
from cntk.contrib import crosstalk as cstk
import tempfile
workdir = tempfile.gettempdir()

data_cntk = [[[1,2,3,4,5],[2,2,3,4,5],[3,2,3,4,5]], [[4,2,3,4,5],[5,2,3,4,5]]]
data_tf   = [[[1,2,3,4,5],[2,2,3,4,5],[3,2,3,4,5]], [[4,2,3,4,5],[5,2,3,4,5],[0,0,0,0,0]]]
data_tf_len = [3,2]
max_seq_len = max(data_tf_len)
batch_size = len(data_tf)

in_dim = 5
dim = 3

def cntk_baseline_lstm():
    import cntk as C
    import cntk.contrib.crosstalk.crosstalk_cntk as crosstalk_cntk
    _ci = crosstalk_cntk.instance
    input_var = C.sequence.input(shape=(in_dim))
    fwbw = C.splice(C.layers.Recurrence(C.layers.LSTM(dim, init_bias=C.glorot_uniform()))(input_var), C.layers.Recurrence(C.layers.LSTM(dim), go_backwards=True)(input_var))
    _ci.watch(fwbw, 'birnn', var_type=cstk.RnnAttr,
          attr=cstk.RnnAttr(bidirectional=True, op_type='lstm', input_dim=in_dim, hidden_dim=dim, forget_bias=0))
    _ci.watch(fwbw, 'birnn_out')

    data = {input_var:data_cntk}
    _ci.set_data(data)
    _ci.set_workdir(workdir)
    _ci.fetch('birnn', save=True)
    _ci.fetch('birnn_out', save=True)
    _ci.reset()

def tf_baseline_lstm():
    import tensorflow as tf # note this test runs with tensorflow 0.12
    import cntk.contrib.crosstalk.crosstalk_tensorflow0_12 as crtf
    _ci = crtf.instance

    tf.reset_default_graph()
    
    with tf.variable_scope("rnn"):
        x = tf.placeholder(tf.float32, [batch_size, max_seq_len, in_dim])
        l = tf.placeholder(tf.int32, [batch_size])
        cell = tf.nn.rnn_cell.BasicLSTMCell(dim)
        (fw, bw), _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, x, l, dtype=tf.float32)
        
        _ci.watch('rnn/BiRNN',
                  'birnn', var_type=cstk.RnnAttr,
                  attr=cstk.RnnAttr(bidirectional=True, op_type='lstm', input_dim=in_dim, hidden_dim=dim, forget_bias=1)) # tf default forget_bias==1

        output = tf.concat(2, [fw, bw])
        _ci.watch(output, 'birnn_out', var_type=crtf.VariableType)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        data = {x:data_tf, l:data_tf_len}
        _ci.set_workdir(workdir)
        _ci.set_data(sess, data)
        _ci.fetch('birnn', save=True)
        _ci.fetch('birnn_out', save=True)
        _ci.reset()
        sess.close()

def test_cntk_cudnn():
    try:
        import tensorflow
        has_tensorflow = True
    except:
        has_tensorflow = False

    if has_tensorflow:
        tf_baseline_lstm()
    else:
        cntk_baseline_lstm()

    import cntk as C
    import cntk.contrib.crosstalk.crosstalk_cntk as crct
    _ci = crct.instance
        
    input_var = C.sequence.input(shape=(in_dim))
    data = {input_var:data_cntk}
    _ci.set_data(data)
    _ci.set_workdir(workdir)

    W = C.parameter((-1,dim,), init=C.glorot_uniform())
    cudnn_fwbw = C.optimized_rnnstack(input_var, W, dim, 1, bidirectional=True, recurrent_op='lstm')
    _ci.watch(cudnn_fwbw, 'cntk_birnn_cudnn', var_type=cstk.RnnAttr,
          attr=cstk.RnnAttr(bidirectional=True, op_type='lstm', input_dim=in_dim, hidden_dim=dim, forget_bias=0))
    _ci.watch(cudnn_fwbw, 'cntk_birnn_cudnn_out')
    
    _ci.assign('cntk_birnn_cudnn', load=True, load_name='cntk_birnn')
    assert _ci.compare('cntk_birnn_cudnn_out', compare_name='cntk_birnn_out')

    _ci.fetch('cntk_birnn_cudnn', save=True)
    _ci.assign('cntk_birnn_cudnn', load=True)
    assert _ci.compare('cntk_birnn_cudnn_out', compare_name='cntk_birnn_out')
    
    _ci.reset()
