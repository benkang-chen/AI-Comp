import keras
from keras import Model
from keras.layers import *
from JoinAttLayer import Attention


class TextClassifier():

    def model(self, embeddings_matrix, maxlen, word_index, num_class):
        inp = Input(shape=(maxlen,))
        encode = Bidirectional(GRU(128, return_sequences=True))
        encode2 = Bidirectional(GRU(128, return_sequences=True))
        attention = Attention(maxlen)
        x_4 = Embedding(len(word_index) + 1,
                        embeddings_matrix.shape[1],
                        weights=[embeddings_matrix],
                        input_length=maxlen,
                        trainable=True)(inp)
        """
            SpatialDropout1D(p)
            SpatialDropout1D与Dropout的作用类似，但它断开的是整个1D特征图，而不是单个神经元。
            如果一张特征图的相邻像素之间有很强的相关性（通常发生在低层的卷积层中），
            那么普通的dropout无法正则化其输出，否则就会导致明显的学习率下降。这种情况下，
            SpatialDropout1D能够帮助提高特征图之间的独立性，应该用其取代普通的Dropout

            参数:
                p：0~1的浮点数，控制需要断开的链接的比例
            输入shape:
                输入形如（samples，timesteps，channels）的3D张量
            
            输出shape:
                与输入相同
        """
        x_3 = SpatialDropout1D(0.2)(x_4)
        x_3 = encode(x_3)
        x_3 = Dropout(0.2)(x_3)
        x_3 = encode2(x_3)
        x_3 = Dropout(0.2)(x_3)
        """
            GlobalAveragePooling1D()
            为时域信号施加全局平均值池化

            输入shape:
                形如（samples，steps，features）的3D张量
            输出shape:
                形如(samples, features)的2D张量
        """
        avg_pool_3 = GlobalAveragePooling1D()(x_3)
        """
            GlobalMaxPooling1D()
            对于时间信号的全局最大池化

            输入shape：
                形如（samples，steps，features）的3D张量
            输出shape：
                形如(samples, features)的2D张量

        """
        max_pool_3 = GlobalMaxPooling1D()(x_3)
        attention_3 = attention(x_3)
        x = keras.layers.concatenate([avg_pool_3, max_pool_3, attention_3], name="fc")
        x = Dense(num_class, activation="sigmoid")(x)

        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,amsgrad=True)
        rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
        model = Model(inputs=inp, outputs=x)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=adam)
        return model
