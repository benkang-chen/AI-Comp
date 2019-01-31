'''This example demonstrates the use of Convolution1D for text classification.

Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb

# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# print(x_train)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
"""
    pad_sequences(sequences, maxlen=None, dtype='int32',
        padding='pre', truncating='pre', value=0.)
        
    将长为nb_samples的序列（标量序列）转化为形如(nb_samples,nb_timesteps)2D numpy array。
    如果提供了参数maxlen，nb_timesteps=maxlen，否则其值为最长序列的长度。
    其他短于该长度的序列都会在后部填充0以达到该长度。长于nb_timesteps的序列将会被截断，以使其匹配目标长度。
    padding和截断发生的位置分别取决于padding和truncating.
    
    参数:
    sequences：浮点数或整数构成的两层嵌套列表
    
    maxlen：None或整数，为序列的最大长度。大于此长度的序列将被截短，小于此长度的序列将在后部填0.
    
    dtype：返回的numpy array的数据类型
    
    padding：‘pre’或‘post’，确定当需要补0时，在序列的起始还是结尾补
    
    truncating：‘pre’或‘post’，确定当需要截断序列时，从起始还是结尾截断
    
    value：浮点数，此值将在填充时代替默认的填充值0
    
    返回值:
    返回形如(nb_samples,nb_timesteps)的2D张量
"""
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
"""
    Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, 
              activity_regularizer=None, embeddings_constraint=None, mask_zero=False, 
              input_length=None)
    将正整数（索引值）转换为固定尺寸的稠密向量。
    参数:
    
    input_dim: int > 0。词汇表大小， 即，最大整数 index + 1。
    output_dim: int >= 0。词向量的维度。
    embeddings_initializer: embeddings 矩阵的初始化方法 (详见 initializers)。
    embeddings_regularizer: embeddings matrix 的正则化方法 (详见 regularizer)。
    embeddings_constraint: embeddings matrix 的约束函数 (详见 constraints)。
    mask_zero: 是否把 0 看作为一个应该被遮蔽的特殊的 "padding" 值。 这对于可变长的 循环神经网络层 十分有用。
               如果设定为 True，那么接下来的所有层都必须支持 masking，否则就会抛出异常。 
               如果 mask_zero 为 True，作为结果，索引 0 就不能被用于词汇表中 
               （input_dim 应该与 vocabulary + 1 大小相同）。
    input_length: 输入序列的长度，当它是固定的时。 如果你需要连接 Flatten 和 Dense 层，
                  则这个参数是必须的 （没有它，dense 层的输出尺寸就无法计算）。
    输入尺寸:
    
    尺寸为 (batch_size, sequence_length) 的 2D 张量。
    
    输出尺寸:
    
    尺寸为 (batch_size, sequence_length, output_dim) 的 3D 张量。
"""
# embedding 层参数的个数为： 5000（词典的大小） * 50 （embedding size）= 250000
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
"""
    Conv1D(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, 
           activation=None, use_bias=True, kernel_initializer='glorot_uniform', 
           bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
           activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
    一维卷积层（即时域卷积），用以在一维输入信号上进行邻域滤波。当使用该层作为首层时，
    需要提供关键字参数input_shape。例如(10,128)代表一个长为10的序列，序列中每个信号为128向量。
    而(None, 128)代表变长的128维向量序列。

    该层生成将输入信号与卷积核按照单一的空域（或时域）方向进行卷积。如果use_bias=True，则还会加上一个偏置项，
    若activation不为None，则输出为经过激活函数的输出。

    参数:
    filters：卷积核的数目（即输出的维度）
    
    kernel_size：整数或由单个整数构成的list/tuple，卷积核的空域或时域窗长度
    
    strides：整数或由单个整数构成的list/tuple，为卷积的步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
    
    padding：补0策略，为“valid”, “same” 或“causal”，“causal”将产生因果（膨胀的）卷积，即output[t]不依赖于input[t+1：]。当对不能违反时间顺序的时序信号建模时有用。参考WaveNet: A Generative Model for Raw Audio, section 2.1.。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
    
    activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
    
    dilation_rate：整数或由单个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
    
    use_bias:布尔值，是否使用偏置项
    
    kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
    
    bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
    
    kernel_regularizer：施加在权重上的正则项，为Regularizer对象
    
    bias_regularizer：施加在偏置向量上的正则项，为Regularizer对象
    
    activity_regularizer：施加在输出上的正则项，为Regularizer对象
    
    kernel_constraints：施加在权重上的约束项，为Constraints对象
    
    bias_constraints：施加在偏置上的约束项，为Constraints对象
    
    输入shape:
    形如（samples，steps，input_dim）的3D张量
    
    输出shape:
    形如（samples，new_steps，nb_filter）的3D张量，因为有向量填充的原因，steps的值会改变
"""
# 该层的参数个数为 3（kernel_size） * 50 （embedding_size） * 250 = 37750
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
"""
    GlobalMaxPooling1D()
    对于时间信号的全局最大池化
    输入shape:
    形如（samples，steps，features）的3D张量
    输出shape:
    形如(samples, features)的2D张量
"""
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
