'''Trains an LSTM model on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.

# Notes

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

max_features = 20000
# cut texts after this number of words (among top max_features most common words)
maxlen = 80
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
"""
    LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, 
         kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
         bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, 
         recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
         kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, 
         recurrent_dropout=0.0)
    参数：
    units：输出维度
    
    activation：激活函数，为预定义的激活函数名（参考激活函数）
    
    recurrent_activation: 为循环步施加的激活函数（参考激活函数）
    
    use_bias: 布尔值，是否使用偏置项
    
    kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。
                        参考initializers
    
    recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。
                          参考initializers
    
    bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。
                      参考initializers
    
    kernel_regularizer：施加在权重上的正则项，为Regularizer对象
    
    bias_regularizer：施加在偏置向量上的正则项，为Regularizer对象
    
    recurrent_regularizer：施加在循环核上的正则项，为Regularizer对象
    
    activity_regularizer：施加在输出上的正则项，为Regularizer对象
    
    kernel_constraints：施加在权重上的约束项，为Constraints对象
    
    recurrent_constraints：施加在循环核上的约束项，为Constraints对象
    
    bias_constraints：施加在偏置上的约束项，为Constraints对象
    
    dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例
    
    recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例
    
    其他参数参考Recurrent的说明
"""
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
