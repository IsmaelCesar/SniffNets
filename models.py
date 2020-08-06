"""
    This module contains the implementations of all three SniffNet models
"""
from keras.layers import Conv2D
from keras.models import Sequential
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Flatten, Dropout, Add
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.layers.merge import concatenate
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN


def sniffnet(input_shape, n_classes):
    kernel = (20, input_shape[1] // 2 - 1)
    multiplier = 10
    out_channels = 5 * multiplier
    # convolutional components
    model = Sequential()
    model.add(Conv2D(out_channels, kernel, input_shape=input_shape, use_bias=True,
                     activation='relu', name='first_conv'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    kernel = ((input_shape[0] - kernel[0] + 1) // 2, kernel[1])
    model.add(Conv2D(out_channels, kernel, use_bias=True,
                     activation='relu', name='second_conv'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(out_channels, use_bias=True, activation='relu', name="camada_fc1"))
    model.add(BatchNormalization())
    model.add(Dense(n_classes, use_bias=True, activation='softmax', name="classificaiton"))

    return model


def sniffresnet(input_shape, n_classes):
    multiplier = 4
    kernel = (8, input_shape[1] // 2 - 1)
    out_channels = 5 * multiplier
    # First Part of the convolution
    x_input = Input(input_shape)
    x_skip = Conv2D(out_channels, kernel,
                    activation='relu', name='first_conv1')(x_input)
    layer_x = Conv2D(out_channels, kernel,
                     padding='same', activation='relu', name='first_conv2')(x_skip)
    layer_x = BatchNormalization()(layer_x)
    layer_x = Add()([layer_x, x_skip])
    layer_x = MaxPooling2D((2, 1), padding='same', name="max_pool1")(layer_x)

    # Second Part of the convolution
    out_channels = out_channels * multiplier
    x_skip = Conv2D(out_channels, kernel, activation='relu', name='second_conv1')(layer_x)
    layer_x = Conv2D(out_channels, kernel, padding='same', use_bias=True,
                     activation='relu', name='second_conv2')(x_skip)
    layer_x = BatchNormalization()(layer_x)
    layer_x = Add()([layer_x, x_skip])
    layer_x = MaxPooling2D((2, 1), name="max_pool2")(layer_x)

    # Fully Connected Part
    layer_x = Flatten()(layer_x)
    layer_x = Dense(100, use_bias=True, activation="relu", name="fc1")(layer_x)
    layer_x = Dropout(.25)(layer_x)
    layer_x = Dense(n_classes, use_bias=True, activation="softmax", name="class")(layer_x)

    model = Model(inputs=x_input, outputs=layer_x, name="SniffResnet")

    return model


def sniffmultinose(input_shape, n_classes):
    inputs_list = []
    multinose_out = []
    for i in range(input_shape[1]):
        x_input = Input((input_shape[0],), name=("input_nose_" + str(i)))
        inputs_list.append(x_input)
        layer_x = Dense(input_shape[0], input_shape=(input_shape[0],),
                        use_bias=True, activation='relu',
                        name=("fc1_nose_" + str(i)))(x_input)
        layer_x = Dense(input_shape[0] // 4, use_bias=True,
                        activation='tanh',
                        name=("fc2_nose_" + str(i)))(layer_x)
        layer_x = Dense(input_shape[0] // 8, use_bias=True,
                        activation='tanh',
                        name=("fc3_nose_" + str(i)))(layer_x)
        multinose_out.append(layer_x)

    concat = concatenate(multinose_out)

    layer_x = Dense(100, activation='tanh', use_bias=True)(concat)
    layer_x = Dense(100, activation='relu', use_bias=True)(layer_x)
    x_out = Dense(n_classes, activation='softmax', name="class")(layer_x)

    model = Model(inputs=inputs_list, outputs=x_out, name="SniffNetMultiNose")
    return model


def get_knn_classifier(n_neighbors):
    return KNN(n_neighbors=3)


def get_svm(m_gamma=8.3):
    return SVC(gamma=m_gamma, C=10, kernel='rbf')


def get_mlp(input_shape, n_classes):
    x_input = Dense(100, input_shape=input_shape, activation='tanh')
    x = Dense(30, activation='tanh')(x_input)
    x = Dense(30, activation='tanh')(x)
    x = Dense(30, activation='tanh')(x)
    x = Dense(30, activation='tanh')(x)
    x = Dense(30, activation='tanh')(x)
    x = Dense(30, activation='tanh')(x)
    x_out = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=x_input, outputs=x_out, name='Simple MLP')
    return model
