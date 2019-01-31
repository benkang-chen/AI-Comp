'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# (60000, 28, 28)
print(x_train.shape)
print(y_train.shape)

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
# (60000, 10)
print(y_train.shape)
y_test = keras.utils.to_categorical(y_test, num_classes)
"""
Dense(output_dim,init='glorot_uniform', activation='linear', weights=None 
W_regularizer=None, b_regularizer=None, activity_regularizer=None, 
W_constraint=None, b_constraint=None, input_dim=None) 
参数：

      output_dim: int >= 0，输出结果的维度
       init : 初始化权值的函数名称或Theano function。可以使用Keras内置的，也可以传递自己编写的Theano function。如果不给weights传递参数时，则该参数必须指明。
       activation : 激活函数名称或者Theano function。可以使用Keras内置的，也可以是传递自己编写的Theano function。如果不明确指定，那么将没有激活函数会被应用。
       weights :用于初始化权值的numpy arrays组成的list。这个List至少有1个元素，其shape为（input_dim, output_dim）。（如果指定init了，那么weights可以赋值None）
       W_regularizer:权值的规则化项，必须传入一个WeightRegularizer的实例（比如L1或L2规则化项。）
       b_regularizer:偏置值的规则化项，必须传入一个WeightRegularizer的实例（比如L1或L2规则化项）。
       activity_regularizer:网络输出的规则化项，必须传入一个ActivityRegularizer的实例。
       W_constraint:权值约束，必须传入一个constraints的实例。
       b_constraint:偏置约束，必须传入一个constraints的实例。
       input_dim:输入数据的维度。这个参数会在模型的第一层中用到。 
"""
model = Sequential()
# 这一层的参数个数 = 784 * 512 + 512(偏置项)
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
# 这一层的参数个数 = 512 * 512 + 512(偏置项)
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
# 这一层的参数个数 = 512 * 10 + 10(偏置项)
model.add(Dense(num_classes, activation='softmax'))

# 打印网络参数
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

"""
fit( x, y, batch_size=32, epochs=10, verbose=1, callbacks=None,
validation_split=0.0, validation_data=None, shuffle=True, 
class_weight=None, sample_weight=None, initial_epoch=0)
参数：
x：输入数据。如果模型只有一个输入，那么x的类型是numpy 
   array，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array

y：标签，numpy array

batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，
   使目标函数优化一步。
   
epochs：整数，训练终止时的epoch值，训练将在达到该epoch值时停止，当没有设置initial_epoch时，
   它就是训练的总轮数，否则训练的总轮数为epochs - inital_epoch
   
verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录

callbacks：list，其中的元素是keras.callbacks.Callback的对象。这个list中的回调函数将会在训练过程
    中的适当时机被调用，参考回调函数
    
validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，
    并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。注意，validation_split的划分在shuffle之前，
    因此如果你的数据本身是有序的，需要先手工打乱再指定validation_split，否则可能会出现验证集样本不均匀。
    
validation_data：形式为（X，y）的tuple，是指定的验证集。此参数将覆盖validation_spilt。

shuffle：布尔值或字符串，一般为布尔值，表示是否在训练过程中随机打乱输入样本的顺序。若为字符串“batch”，
    则是用来处理HDF5数据的特殊情况，它将在batch内部将数据打乱。
    
class_weight：字典，将不同的类别映射为不同的权值，该参数用来在训练过程中调整损失函数（只能用于训练）

sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。
    可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，
    传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。
    这种情况下请确定在编译模型时添加了sample_weight_mode=’temporal’。

initial_epoch: 从该参数指定的epoch开始训练，在继续之前的训练时有用。
"""
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
"""
"""
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
