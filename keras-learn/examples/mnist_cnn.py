'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 返回默认的图像的维度顺序（‘channels_last’或‘channels_first’）
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
"""
keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', 
                    data_format=None, dilation_rate=(1, 1), activation=None, 
                    use_bias=True, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, 
                    bias_regularizer=None, activity_regularizer=None, 
                    kernel_constraint=None, bias_constraint=None)
                    
该层创建了一个卷积核， 该卷积核对层输入进行卷积， 以生成输出张量。 如果 use_bias 为 True， 
则会创建一个偏置向量并将其添加到输出中。 最后，如果 activation 不是 None，它也会应用于输出。
当使用该层作为模型第一层时，需要提供 input_shape 参数 （整数元组，不包含样本表示的轴），
例如， input_shape=(128, 128, 3) 表示 128x128 RGB 图像， 在 data_format="channels_last" 时。

参数:
filters: 整数，输出空间的维度 （即卷积中滤波器的输出数量）。
kernel_size: 一个整数，或者 2 个整数表示的元组或列表， 指明 2D 卷积窗口的宽度和高度。 
            可以是一个整数，为所有空间维度指定相同的值。
strides: 一个整数，或者 2 个整数表示的元组或列表， 指明卷积沿宽度和高度方向的步长。 可以是一个整数，
         为所有空间维度指定相同的值。 指定任何 stride 值 != 1 与指定 dilation_rate 值 != 1 两者不兼容。
padding: "valid" 或 "same" (大小写敏感)。
data_format: 字符串，  channels_last (默认) 或 channels_first 之一，表示输入中维度的顺序。 
             channels_last 对应输入尺寸为 (batch, height, width, channels)，  
             channels_first 对应输入尺寸为 (batch, channels, height, width)。 
             它默认为从 Keras 配置文件 ~/.keras/keras.json 中 找到的 image_data_format 值。
             如果你从未设置它，将使用 channels_last。
dilation_rate: 一个整数或 2 个整数的元组或列表， 指定膨胀卷积的膨胀率。 可以是一个整数，
               为所有空间维度指定相同的值。 当前，指定任何 dilation_rate 值 != 1 与 指定 stride 值 != 1 
               两者不兼容。
activation: 要使用的激活函数 (详见 activations)。 如果你不指定，则不使用激活函数 (即线性激活： a(x) = x)。
use_bias: 布尔值，该层是否使用偏置向量。
kernel_initializer: kernel 权值矩阵的初始化器 (详见 initializers)。
bias_initializer: 偏置向量的初始化器 (详见 initializers)。
kernel_regularizer: 运用到 kernel 权值矩阵的正则化函数 (详见 regularizer)。
bias_regularizer: 运用到偏置向量的正则化函数 (详见 regularizer)。
activity_regularizer: 运用到层输出（它的激活值）的正则化函数 (详见 regularizer)。
kernel_constraint: 运用到 kernel 权值矩阵的约束函数 (详见 constraints)。
bias_constraint: 运用到偏置向量的约束函数 (详见 constraints)。
输入尺寸:
如果 data_format='channels_first'， 输入 4D 张量，尺寸为 (samples, channels, rows, cols)。
如果 data_format='channels_last'， 输入 4D 张量，尺寸为 (samples, rows, cols, channels)。
输出尺寸:
如果 data_format='channels_first'， 输出 4D 张量，尺寸为 (samples, filters, new_rows, new_cols)。
如果 data_format='channels_last'， 输出 4D 张量，尺寸为 (samples, new_rows, new_cols, filters)。
由于填充的原因， rows 和 cols 值可能已更改。
"""
# 第一层参数的个数 = （3 * 3 *1）[卷积核的大小] * 32[卷积核个数] + 32[偏置项的个数=卷积核个数]
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# 第二层参数个数 = （3 * 3 * 32）[卷积核大小] * 64[卷积核个数] + 64[偏置项个数]
model.add(Conv2D(64, (3, 3), activation='relu'))
"""
MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
为空域信号施加最大值池化
参数:
pool_size：整数或长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，
           如取（2，2）将使图片在两个维度上均变为原长的一半。为整数意为各个维度值相同且为该数字。

strides：步长值, 整数或长为2的整数tuple，或者None，如果为None将会默认为pool_size。

border_mode：‘valid’或者‘same’

data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。
             该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，
             “channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为
             （3,128,128），而“channels_last”应将数据组织为（128,128,3）。
             该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。

输入shape:
‘channels_first’模式下，为形如（samples，channels, rows，cols）的4D张量

‘channels_last’模式下，为形如（samples，rows, cols，channels）的4D张量

输出shape:
‘channels_first’模式下，为形如（samples，channels, pooled_rows, pooled_cols）的4D张量

‘channels_last’模式下，为形如（samples，pooled_rows, pooled_cols，channels）的4D张量
"""
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
