'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
from __future__ import print_function

import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

max_words = 1000
batch_size = 32
epochs = 5

print('Loading data...')
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                         test_split=0.2)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

print('Vectorizing sequence data...')
"""
    Tokenizer(num_words=None,
                                   filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                                   lower=True,
                                   split=" ",
                                   char_level=False)
    Tokenizer是一个用于向量化文本，或将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）的类。
    构造参数:
    与text_to_word_sequence同名参数含义相同
    
    num_words：None或整数，处理的最大单词数量。若被设置为整数，
               则分词器将被限制为待处理数据集中最常见的num_words个单词
    
    char_level: 如果为 True, 每个字符将被视为一个标记
    类方法:
    fit_on_texts(texts)
    
    texts：要用以训练的文本列表
    texts_to_sequences(texts)
    
    texts：待转为序列的文本列表
    
    返回值：序列的列表，列表中每个序列对应于一段输入文本
    
    texts_to_sequences_generator(texts)
    
    本函数是texts_to_sequences的生成器函数版
    
    texts：待转为序列的文本列表
    
    返回值：每次调用返回对应于一段输入文本的序列
    
    texts_to_matrix(texts, mode)：
    
    texts：待向量化的文本列表
    
    mode：‘binary’，‘count’，‘tfidf’，‘freq’之一，默认为‘binary’
    
    返回值：形如(len(texts), nb_words)的numpy array
    
    fit_on_sequences(sequences):
    
    sequences：要用以训练的序列列表
    sequences_to_matrix(sequences):
    
    sequences：待向量化的序列列表
    
    mode：‘binary’，‘count’，‘tfidf’，‘freq’之一，默认为‘binary’
    
    返回值：形如(len(sequences), nb_words)的numpy array
    属性
    word_counts:字典，将单词（字符串）映射为它们在训练期间出现的次数。仅在调用fit_on_texts之后设置。
    word_docs: 字典，将单词（字符串）映射为它们在训练期间所出现的文档或文本的数量。
               仅在调用fit_on_texts之后设置。
    word_index: 字典，将单词（字符串）映射为它们的排名或者索引。仅在调用fit_on_texts之后设置。
    document_count: 整数。分词器被训练的文档（文本或者序列）数量。
                    仅在调用fit_on_texts或fit_on_sequences之后设置。
"""
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print('Building model...')
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
