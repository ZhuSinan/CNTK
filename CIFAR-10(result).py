import os
import numpy as np
import matplotlib.pyplot as plt
import math
import cntk as C
from cntk.layers import default_options, Convolution, MaxPooling, AveragePooling, Dropout, BatchNormalization, Dense,\
    Sequential, For
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
import cntk.io.transforms as xforms
from cntk.initializer import glorot_uniform, he_normal
from cntk import Trainer
from cntk.learners import momentum_sgd, learning_rate_schedule, UnitType, momentum_as_time_constant_schedule
from cntk import cross_entropy_with_softmax, classification_error, relu, input, softmax, element_times
from cntk.ops import combine, times, element_times, AVG_POOLING
from cntk.logging import *

#C.device.set_default_device(C.device.gpu(0))
C.DeviceDescriptor.try_set_default_device(C.device.gpu(0))

# 1. basic model
def create_basic_model(input, out_dims):
    net = Convolution((5,5), 32, init=glorot_uniform(), activation=relu, pad=True)(input)
    net = MaxPooling((3,3), strides=(2,2))(net)

    net = Convolution((5,5), 64, init=glorot_uniform(), activation=relu, pad=True)(net)
    net = MaxPooling((3,3), strides=(2,2))(net)

    net = Convolution((5,5), 128, init=glorot_uniform(), activation=relu, pad=True)(net)
    net = MaxPooling((3,3), strides=(2,2))(net)

    net = Dense(256, init=glorot_uniform())(net)
    net = Dense(out_dims, init=glorot_uniform(), activation=None)(net)

    return net

# 2. model with dropout
def creat_basic_model_with_dropout(input, out_dims):
    net = Convolution((5, 5), 32, init=glorot_uniform(), activation=relu, pad=True)(input)
    net = MaxPooling((3, 3), strides=(2, 2))(net)

    net = Convolution((5, 5), 64, init=glorot_uniform(), activation=relu, pad=True)(net)
    net = MaxPooling((3, 3), strides=(2, 2))(net)

    net = Convolution((5, 5), 128, init=glorot_uniform(), activation=relu, pad=True)(net)
    net = MaxPooling((3, 3), strides=(2, 2))(net)

    net = Dense(256, init=glorot_uniform())(net)
    Dropout(0.2)
    net = Dense(out_dims, init=glorot_uniform(), activation=None)(net)

    return net

# 3. model with batch normalization
def creat_basic_model_with_batch_normalization(input, out_dims):
    net = Convolution((5, 5), 32, init=glorot_uniform(), activation=relu, pad=True)(input)
    net = MaxPooling((3, 3), strides=(2, 2))(net)

    net = Convolution((5, 5), 64, init=glorot_uniform(), activation=relu, pad=True)(net)
    net = MaxPooling((3, 3), strides=(2, 2))(net)

    net = Convolution((5, 5), 128, init=glorot_uniform(), activation=relu, pad=True)(net)
    net = MaxPooling((3, 3), strides=(2, 2))(net)

    net = Dense(256, init=glorot_uniform())(net)
    BatchNormalization(map_rank=1)
    #Dropout(0.2)
    net = Dense(out_dims, init=glorot_uniform(), activation=None)(net)

    return net

# 4. model of vgg
def create_vgg9_model(input, out_dims):
    with default_options(activation=relu):
        model = Sequential([
            For(range(3), lambda i: [
                Convolution((3, 3), [64, 96, 128][i], init=glorot_uniform(), pad=True),
                Convolution((3, 3), [64, 96, 128][i], init=glorot_uniform(), pad=True),
                MaxPooling((3, 3), strides=(2, 2))
            ]),
            For(range(2), lambda: [
                Dense(1024, init=glorot_uniform())
            ]),
            Dense(out_dims, init=glorot_uniform(), activation=None)
        ])

    return model(input)


# 5. model of resNet
def convolution_bn(input, filter_size, num_filters, strides=(1, 1), init=he_normal(), activation=relu):
    if activation is None:
        activation = lambda x: x

    r = Convolution(filter_size, num_filters, strides=strides, init=init, activation=None, pad=True, bias=False)(input)
    r = BatchNormalization(map_rank=1)(r)
    r = activation(r)

    return r


def resnet_basic(input, num_filters):
    c1 = convolution_bn(input, (3, 3), num_filters)
    c2 = convolution_bn(c1, (3, 3), num_filters, activation=None)
    p = c2 + input
    return relu(p)


def resnet_basic_inc(input, num_filters):
    c1 = convolution_bn(input, (3, 3), num_filters, strides=(2, 2))
    c2 = convolution_bn(c1, (3, 3), num_filters, activation=None)

    s = convolution_bn(input, (1, 1), num_filters, strides=(2, 2), activation=None)

    p = c2 + s
    return relu(p)


def resnet_basic_stack(input, num_filters, num_stack):
    assert (num_stack > 0)

    r = input
    for _ in range(num_stack):
        r = resnet_basic(r, num_filters)
    return r


def create_resnet_model(input, out_dims):
    conv = convolution_bn(input, (3, 3), 16)
    r1_1 = resnet_basic_stack(conv, 16, 3)

    r2_1 = resnet_basic_inc(r1_1, 32)
    r2_2 = resnet_basic_stack(r2_1, 32, 2)

    r3_1 = resnet_basic_inc(r2_2, 64)
    r3_2 = resnet_basic_stack(r3_1, 64, 2)

    # Global average pooling
    pool = AveragePooling(filter_shape=(8, 8), strides=(1, 1))(r3_2)
    net = Dense(out_dims, init=he_normal(), activation=None)(pool)

    return net



image_height = 32
image_width  = 32
num_channels = 3
num_classes  = 10


def create_reader(map_file, mean_file, train):
    transforms = []
    if train:
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=0.8)
        ]
    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        xforms.mean(mean_file)
    ]

    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features = StreamDef(field='image', transforms=transforms),
        labels = StreamDef(field='label', shape=num_classes)
    )))


def train_and_evaluate(reader_train, read_test, max_epochs, model_func):
    input_var = C.input((num_channels, image_height, image_width))
    label_var = C.input((num_classes))

    feature_scale = 1 / 256.0
    input_var_norm = element_times(feature_scale, input_var)

    z = model_func(input_var_norm, out_dims = 10)

    ce = cross_entropy_with_softmax(z, label_var)
    pe = classification_error(z, label_var)

    epoch_size = 50000
    minibatch_size = 64

    lr_per_minibatch = learning_rate_schedule([0.01]*10 + [0.003]*10 + [0.001], UnitType.minibatch, epoch_size)
    momentum_time_constant = momentum_as_time_constant_schedule(-minibatch_size/np.log(0.9))
    l2_reg_weight = 0.001

    learner = momentum_sgd(z.parameters,
                           lr=lr_per_minibatch, momentum=momentum_time_constant,
                           l2_regularization_weight=l2_reg_weight)
    progress_printer = ProgressPrinter(tag='Training', num_epochs=max_epochs)
    trainer = Trainer(z, (ce, pe), [learner], [progress_printer])

    input_map = {
        input_var : reader_train.streams.features,
        label_var : reader_train.streams.labels
    }

    log_number_of_parameters(z)
    print()

    batch_index = 0
    plot_data = {'batchindex':[], 'loss':[], 'error':[]}
    for epoch in range(max_epochs):
        sample_count = 0
        while sample_count < epoch_size:
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=input_map)
            trainer.train_minibatch(data)

            sample_count += data[label_var].num_samples

            plot_data['batchindex'].append(batch_index)
            plot_data['loss'].append(trainer.previous_minibatch_loss_average)
            plot_data['error'].append(trainer.previous_minibatch_evaluation_average)

            batch_index += 1
        trainer.summarize_training_progress()

    epoch_size = 10000
    minibatch_size = 16

    metric_numer = 0
    metric_denom = 0
    sample_count = 0
    minibatch_index = 0

    while sample_count < epoch_size:
        current_minibatch = min(minibatch_size, epoch_size - sample_count)

        data = read_test.next_minibatch(current_minibatch, input_map=input_map)

        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch

        # Keep track of the number of samples processed so far.
        sample_count += data[label_var].num_samples
        minibatch_index += 1

    print("")
    print("Final Results: Minibatch[1-{}]: accuracy = {:0.1f}% ".format(minibatch_index + 1,
                                                                        100 - (metric_numer )* 100.0 / metric_denom))

    print("")

    # Visualize training result:
    window_width = 32
    loss_cumsum = np.cumsum(np.insert(plot_data['loss'], 0, 0))
    error_cumsum = np.cumsum(np.insert(plot_data['error'], 0, 0))

    # Moving average.
    plot_data['batchindex'] = np.insert(plot_data['batchindex'], 0, 0)[window_width:]
    plot_data['avg_loss'] = (loss_cumsum[window_width:] - loss_cumsum[:-window_width]) / window_width
    plot_data['avg_error'] = (error_cumsum[window_width:] - error_cumsum[:-window_width]) / window_width

    plt.figure(1)
    plt.subplot(211)
    plt.plot(plot_data["batchindex"], plot_data["avg_loss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss ')

    plt.show()

    plt.subplot(212)
    plt.plot(plot_data["batchindex"], plot_data["avg_error"], 'r--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Label Prediction Error')
    plt.title('Minibatch run vs. Label Prediction Error ')
    plt.show()

    return softmax(z)

data_path = os.path.join('data', 'CIFAR-10')
reader_train = create_reader(os.path.join(data_path, 'train_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'),
                             True)
reader_test = create_reader(os.path.join(data_path, 'test_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'),
                            False)
#pred = train_and_evaluate(reader_train, reader_test, max_epochs=30, model_func=create_basic_model)

pred_dropout = train_and_evaluate(reader_train, reader_test, max_epochs=30, model_func=creat_basic_model_with_dropout)

pred_batch_normalization = train_and_evaluate(reader_train, reader_test, max_epochs=30, model_func=creat_basic_model_with_batch_normalization)

pred_vgg = train_and_evaluate(reader_train, reader_test, max_epochs=30, model_func=create_vgg9_model)

pre_res = train_and_evaluate(reader_train, reader_test, max_epochs=30, model_func=create_resnet_model)



