from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import Conv1D, GlobalMaxPool1D, MaxPool1D, Dropout, Concatenate
import tensorflow as tf

def CNN(input_dim,
        input_length,
        vec_size,
        output_shape):
    '''
    Creat CNN net,use Embedding+CNN1D+GlobalMaxPool1D+Dense.
    You can change filters and dropout rate in code..

    :param input_dim: Size of the vocabulary
    :param input_length:Length of input sequences
    :param vec_size:Dimension of the dense embedding
    :param output_shape:Target shape,target should be one-hot term
    :return:keras model
    '''
    data_input = Input(shape=[input_length])
    word_vec = Embedding(input_dim=input_dim + 1,
                         input_length=input_length,
                         output_dim=vec_size)(data_input)
    cnn1 = Conv1D(filters=256,
               kernel_size=[3],
               strides=1,
               padding='same',
               activation='relu')(word_vec)
    cnn1 = GlobalMaxPool1D()(cnn1)
    cnn2 = Conv1D(filters=256,
               kernel_size=[6],
               strides=1,
               padding='same',
               activation='relu')(word_vec)
    cnn2 = GlobalMaxPool1D()(cnn2)
    cnn3 = Conv1D(filters=256,
               kernel_size=[9],
               strides=1,
               padding='same',
               activation='relu')(word_vec)
    cnn3 = GlobalMaxPool1D()(cnn3)
    x = Concatenate()([cnn1,cnn2,cnn3])
    x = Dense(500, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(output_shape, activation='sigmoid')(x)
    model = Model(inputs=data_input, outputs=x)
    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['acc'])
    return model


if __name__ == '__main__':
    model = CNN(input_dim=10, input_length=10, vec_size=10, output_shape=10)
    model.summary()
