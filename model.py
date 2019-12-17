import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Add, BatchNormalization, Conv1D, Dense, Dropout, \
                                    Flatten, Input, Lambda, MaxPooling1D, \
                                    ReLU, Softmax, TimeDistributed
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers


def ctc_lambda_func(args):
    '''
    y_true = numeric translation of text
    y_pred = output of softmax layer
    input_length = output sequence length
    label_length = length of the true sequence
    '''
    y_true, y_pred, input_length, label_length = args
    print(y_true.shape)
    print(y_pred.shape)
    print(input_length)
    print(label_length)
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)


def res_block(X, filters, k_width, dropout, block_num, block_depth, layer_num):
    """
    X: input from previous layer/block
    filters: filter length
    k_size: width of kernel
    block_num: the block number
    block_depth: the total depth of the block, how many conv blocks to stack before adding residual

    """
    # Defining name basis
    conv_name_base = 'res' + str(block_num) + '_' + str(block_depth) + '_' + str(layer_num)
    bn_name_base = 'bn' + str(block_num) + '_' + str(block_depth) + '_' + str(layer_num)
    skip_name_base = 'skip' + str(block_num) + '_' + str(block_depth) + '_' + str(layer_num)

    # Save the input value
    X_shortcut = X
    X_shortcut = Conv1D(filters, 1, strides=1, padding='same', dilation_rate=1,
                        kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01),
                        name=skip_name_base)(X_shortcut)
    X_shortcut = BatchNormalization(momentum=0.997, name=bn_name_base + '_D')(X_shortcut)

    # Define residual block construction loop
    i = 0
    while i < block_depth:
        if i == block_depth-1:
            X = Conv1D(filters, k_width, strides=1, padding='same', dilation_rate=1, kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(0.01), name=conv_name_base + '_r' + str(i))(X)
            X = BatchNormalization(momentum=0.997, name=bn_name_base + '_r' + str(i))(X)
            break
        X = Conv1D(filters, k_width, strides=1, padding='same', dilation_rate=1, kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(0.01), name=conv_name_base + '_r' + str(i))(X)
        X = BatchNormalization(momentum=0.997, name=bn_name_base + '_r' + str(i))(X)
        X = ReLU()(X)
        X = Dropout(dropout)(X)
        i += 1
    X += X_shortcut
    X = ReLU()(X)
    X = Dropout(dropout)(X)
    return X


def large_model(input_shape, label_num, max_text_len, train=True):
    """
    input_shape: as batching is used, should be of shape (None, j, k) where j = number of time steps
                 and k = number of features per time step
    label_num: a dictionary of characters to be used in calculating CTC loss
    max_text_length: the maximum length of the ground truth sentences
    train: boolean for training or not, defines the input and output structure of the model
    """
    X_input = Input(batch_shape=input_shape, name='input')

    # First conv block
    X = Conv1D(256, 11, strides=1, padding='same', dilation_rate=1, kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(0.01), name='conv_input')(X_input)
    X = BatchNormalization(momentum=0.997)(X)
    # Use relu activation function
    X = ReLU()(X)
    X = Dropout(0.20)(X)
    
    # Add residual blocks
    X = res_block(X, 256, 11, 0.2, 1, 5, 0)
    X = res_block(X, 256, 11, 0.2, 1, 5, 1)

    X = res_block(X, 384, 13, 0.2, 2, 5, 0)
    X = res_block(X, 384, 13, 0.2, 2, 5, 1)

    X = res_block(X, 512, 17, 0.2, 3, 5, 0)
    X = res_block(X, 512, 17, 0.2, 3, 5, 1)

    X = res_block(X, 640, 21, 0.3, 4, 5, 0)
    X = res_block(X, 640, 21, 0.3, 4, 5, 1)

    X = res_block(X, 768, 25, 0.3, 5, 5, 0)
    X = res_block(X, 768, 25, 0.3, 5, 5, 1)

    # Final conv layers
    X = Conv1D(896, 29, strides=1, padding='same', dilation_rate=2,
               kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01),
               name='conv_dil2')(X)
    X = BatchNormalization(momentum=0.997)(X)
    X = ReLU()(X)
    X = Dropout(0.40)(X)

    X = Conv1D(1024, 1, strides=1, padding='same', dilation_rate=1,
               kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01),
               name='conv_end1')(X)
    X = BatchNormalization(momentum=0.997)(X)
    X = ReLU()(X)
    X = Dropout(0.40)(X)

    # Final fc layer with softmax output
    X = Conv1D(label_num, 1, strides=1, padding='same', dilation_rate=1,
               kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01),
               name='conv_end2')(X)
    y_pred = Softmax(name='softmax')(X)

    if train:
        # Section regarding the implementation of CTC loss function
        labels = Input(name='the_labels', shape=(max_text_len,), dtype='int16')
        input_length = Input(name='input_length', shape=(1,), dtype='int16')
        label_length = Input(name='label_length', shape=(1,), dtype='int16')

        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([labels, y_pred, input_length, label_length]) #(None, 1)

        model = Model(inputs=[X_input, labels, input_length, label_length], outputs=loss_out)

        # Clipnorm seems to speeds up convergence
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

        # The loss calc occurs elsewhere, so use a dummy lambda func for the loss
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    else:
        model = Model(inputs=X_input, outputs=y_pred)

    return model


def small_model(input_shape, label_num, max_text_len, train=True):
    """
    input_shape: as batching is used, should be of shape (None, j, k) where j = number of time steps
                 and k = number of features per time step
    label_num: a dictionary of characters to be used in calculating CTC loss
    max_text_length: the maximum length of the ground truth sentences
    train: boolean for training or not, defines the input and output structure of the model
    """
    X_input = Input(batch_shape=input_shape, name='input')

    # First conv block
    X = Conv1D(256, 11, strides=1, padding='same', dilation_rate=1, kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(0.01), name='conv_input')(X_input)
    X = BatchNormalization(momentum=0.997)(X)
    # Use relu activation function
    X = ReLU()(X)
    X = Dropout(0.20)(X)

    # Add residual blocks
    X = res_block(X, 256, 11, 0.2, 1, 3, 0)

    X = res_block(X, 384, 13, 0.2, 2, 3, 0)

    X = res_block(X, 512, 17, 0.2, 3, 3, 0)

    X = res_block(X, 640, 21, 0.3, 4, 3, 0)

    X = res_block(X, 768, 25, 0.3, 5, 3, 0)

    # Final conv layers
    X = Conv1D(896, 29, strides=1, padding='same', dilation_rate=2,
               kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01),
               name='conv_dil2')(X)
    X = BatchNormalization(momentum=0.997)(X)
    X = ReLU()(X)
    X = Dropout(0.40)(X)

    X = Conv1D(1024, 1, strides=1, padding='same', dilation_rate=1,
               kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01),
               name='conv_end1')(X)
    X = BatchNormalization(momentum=0.997)(X)
    X = ReLU()(X)
    X = Dropout(0.40)(X)

    # Final fc layer with softmax output
    X = Conv1D(label_num, 1, strides=1, padding='same', dilation_rate=1,
               kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01),
               name='conv_end2')(X)
    y_pred = Softmax(name='softmax')(X)

    if train:
        # Section regarding the implementation of CTC loss function
        labels = Input(name='the_labels', shape=(max_text_len,), dtype='int16')
        input_length = Input(name='input_length', shape=(1,), dtype='int16')
        label_length = Input(name='label_length', shape=(1,), dtype='int16')

        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
            [labels, y_pred, input_length, label_length])  # (None, 1)

        model = Model(inputs=[X_input, labels, input_length, label_length], outputs=loss_out)

        # Clipnorm seems to speeds up convergence
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

        # The loss calc occurs elsewhere, so use a dummy lambda func for the loss
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    else:
        model = Model(inputs=X_input, outputs=y_pred)

    return model