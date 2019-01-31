'''Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print('y_train_shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
# print(y_train)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
# 该层参数个数 = （3 * 3 * 3）[卷积核大小] * 32[卷积核个数] + 32[偏置项参数个数]
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
# 激活函数可以通过设置单独的激活层实现，也可以在构造层对象时通过传递activation参数实现。
model.add(Activation('relu'))
# 该层参数个数 = （3 * 3 * 32）[卷积核大小] * 32[卷积核个数] + 32[偏置项参数个数]
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 该层参数个数 = （3 * 3 * 32）[卷积核大小] * 64[卷积核个数] + 64[偏置项参数个数]
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
# 该层参数个数 = （3 * 3 * 64）[卷积核大小] * 64[卷积核个数] + 64[偏置项参数个数]
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.summary()
# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # 布尔值，使输入数据集去中心化（均值为0）, 按feature执行
        featurewise_center=False,  # set input mean to 0 over the dataset
        # 布尔值，使输入数据的每个样本均值为0
        samplewise_center=False,  # set each sample mean to 0
        # 布尔值，将输入除以数据集的标准差以完成标准化, 按feature执行
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        # 布尔值，将输入的每个样本除以其自身的标准差
        samplewise_std_normalization=False,  # divide each input by its std
        # 布尔值，对输入数据施加ZCA白化
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        # 整数，数据提升时图片随机转动的角度
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # 浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # 浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        # 浮点数，剪切强度（逆时针方向的剪切变换角度）
        shear_range=0.,  # set range for random shear
        # 浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，
        # 则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
        zoom_range=0.,  # set range for random zoom
        # 浮点数，随机通道偏移的幅度
        channel_shift_range=0.,  # set range for random channel shifts
        # ‘constant’，‘nearest’，‘reflect’或‘wrap’之一，当进行变换时超出边界的
        # 点将根据本参数给定的方法进行处理
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # 浮点数或整数，当fill_mode=constant时，指定要向超出边界的点填充的值
        cval=0.,  # value used for fill_mode = "constant"
        # 布尔值，进行随机水平翻转
        horizontal_flip=True,  # randomly flip images
        # 布尔值，进行随机竖直翻转
        vertical_flip=False,  # randomly flip images
        # 重放缩因子,默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前)
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    """
    图像生成器方法：
        fit(X, augment=False, rounds=1)：计算依赖于数据的变换所需要的统计信息(均值方差等),
            只有使用featurewise_center，featurewise_std_normalization或zca_whitening时需要此函数。
    参数：
        X：numpy array，样本数据，秩应为4.在黑白图像的情况下channel轴的值为1，在彩色图像情况下值为3
        augment：布尔值，确定是否使用随即提升过的数据
        round：若设augment=True，确定要在数据上进行多少轮数据提升，默认值为1
        seed: 整数,随机数种子
    """
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    """
    图像生成器方法：
    flow(self, X, y, batch_size=32, shuffle=True, seed=None, 
         save_to_dir=None, save_prefix='', save_format='jpeg')：
         接收numpy数组和标签为参数,生成经过数据提升或标准化后的batch数据,
         并在一个无限循环中不断的返回batch数据
    参数：
    X：样本数据，秩应为4.在黑白图像的情况下channel轴的值为1，在彩色图像情况下值为3

    y：标签

    batch_size：整数，默认32

    shuffle：布尔值，是否随机打乱数据，默认为True

    save_to_dir：None或字符串，该参数能让你将提升后的图片保存起来，用以可视化

    save_prefix：字符串，保存提升后图片时使用的前缀, 仅当设置了save_to_dir时生效

    save_format："png"或"jpeg"之一，指定保存图片的数据格式,默认"jpeg"

    yields:形如(x,y)的tuple,x是代表图像数据的numpy数组.y是代表标签的numpy数组.该迭代器无限循环.

    seed: 整数,随机数种子
    """
    #
    """
    fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None,
                  validation_data=None, validation_steps=None, class_weight=None, max_q_size=10, 
                  workers=1, pickle_safe=False, initial_epoch=0)
    利用Python的生成器，逐个生成数据的batch并进行训练。生成器与模型将并行执行以提高效率。
    例如，该函数允许我们在CPU上进行实时的数据提升，同时在GPU上进行模型训练
    函数的参数是：
        generator：生成器函数，生成器的输出应该为：
                   一个形如（inputs，targets）的tuple
                   一个形如（inputs, targets,sample_weight）的tuple。
                   所有的返回值都应该包含相同数目的样本。生成器将无限在数据集上循环。
                   每个epoch以经过模型的样本数达到samples_per_epoch时，记一个epoch结束

        steps_per_epoch：整数，当生成器返回steps_per_epoch次数据时计一个epoch结束，执行下一个epoch
                         建议值为样本总量除以train_flow的batch_size.

        epochs：整数，数据迭代的轮数

        verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录

        validation_data：具有以下三种形式之一
             生成验证集的生成器： 
                 一个形如（inputs,targets）的tuple
                 一个形如（inputs,targets，sample_weights）的tuple
                 validation_steps: 当validation_data为生成器时，本参数指定验证集的生成器返回次数

        class_weight：规定类别权重的字典，将类别映射为权重，常用于处理样本不均衡问题。

        sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与
                       样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为
                       （samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。
                       这种情况下请确定在编译模型时添加了sample_weight_mode='temporal'。

        workers：最大进程数

        max_q_size：生成器队列的最大容量

        pickle_safe: 若为真，则使用基于进程的线程。由于该实现依赖多进程，
                     不能传递non picklable（无法被pickle序列化）的参数到生成器中，
                     因为无法轻易将它们传入子进程中。

        initial_epoch: 从该参数指定的epoch开始训练，在继续之前的训练时有用。
    """
    # flow一个返回的是batch_size个数据，一个batch_size的数据更新一次梯度
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4,
                        steps_per_epoch=1875)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
