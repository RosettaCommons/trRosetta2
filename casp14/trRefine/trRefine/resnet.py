import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np

# Constructs a graph of resnet block
# Default input is channles last.
def resnet_block(_input,
                 isTraining,
                 channel=128,
                 inst_norm=True, # use instance normalization or not
                 p_dropout=None,
                 dilation_rate=(1,1),
                 data_format="channels_last"):
    
    if channel%2 != 0:
        print("Even number channels are required.")
        return -1
    down_channel = channel/2
    
    # InstNorm if needed
    if inst_norm: _input = tf.contrib.layers.instance_norm(_input)

    # Non linearity
    _input = tf.nn.elu(_input)
    # Projection down to 64 dims
    _input = tf.layers.conv2d(_input,
                              filters=down_channel,
                              kernel_size=(1,1),
                              data_format=data_format)
    
    # BatchNorm if needed
    if inst_norm: _input = tf.contrib.layers.instance_norm(_input)
    
    # Non linearity
    _input = tf.nn.elu(_input)
    # 3 by 3 dialated convlolution with increasing dialation rate.
    _input = tf.layers.conv2d(_input,
                              filters=down_channel, 
                              kernel_size=(3,3), 
                              dilation_rate=dilation_rate,
                              data_format=data_format,
                              padding="same")
    
    # BatchNorm if needed
    if inst_norm: _input = tf.contrib.layers.instance_norm(_input)
    
    # Non linearity
    _input = tf.nn.elu(_input)

    # add dropout if p_dropout != None
    if p_dropout != None: _input = tf.layers.dropout(_input, rate=p_dropout, training=isTraining)

    # Projection up to 128 dims.
    _input = tf.layers.conv2d(_input,
                              filters=channel, 
                              kernel_size=(1,1),
                              data_format=data_format)
    return _input

# Creates a resnet architecture.
def build_resnet(_input,
                 channel,
                 num_chunks,
                 isTraining,
                 p_dropout=None,
                 inst_norm=True, #Whether you need bn or not.
                 data_format="channels_last",
                 first_projection=True,
                 final_channel=None,
                 dr_s=[1,2,4,8],
                 ):
   
    # Projection of the very first input to 128 channels.
    if first_projection:
        _input = tf.layers.conv2d(_input,
                                  filters=channel,
                                  kernel_size=(1,1),
                                  dilation_rate=(1,1),
                                  data_format=data_format)
    
    # each chunk contatins 4 blocks with cycling dilation rates.
    for i in range(num_chunks):
        # dilation rates
        for dr in dr_s:
            # save residual connection
            _residual = _input
            # pass through resnet block
            _conved = resnet_block(_input,
                                   isTraining,
                                   p_dropout=p_dropout,
                                   channel=channel,
                                   dilation_rate=(dr, dr),
                                   inst_norm=inst_norm,
                                   data_format=data_format)
            # genearte input to the next block
            _input = _residual+_conved

    if final_channel != None:
        # add 2 more blocks then reduce to the final channel dimmension
        for i in range(2):
            _residual = _input
            _conved = resnet_block(_input,
                                   isTraining,
                                   p_dropout=p_dropout,
                                   channel=channel,
                                   dilation_rate=(1, 1),
                                   inst_norm=inst_norm,
                                   data_format=data_format)
            _input = _residual + _conved
        #
        _input = tf.layers.conv2d(_input,
                                  filters=final_channel,
                                  kernel_size=(1,1),
                                  dilation_rate=(1,1),
                                  data_format=data_format)
            
    return _input
