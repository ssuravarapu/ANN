import time
from keras import models
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, Sequential
from keras.applications import vgg16
from keras.datasets import cifar10
from keras import utils
from keras.layers import Dense, Input, Flatten, Dropout


def pretrained_cifar10():
    dir_path = "/Users/surya/ai/trained_models/"
    model_name = 'cifar10_cnn.h5'
    model_path = dir_path + model_name
    num_classes = 10
    batch_size = 64
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_test = x_test / 255
    y_test = utils.to_categorical(y_test, num_classes)
    model = models.load_model(model_path)
    print("Model Summary: " + "\n" + str(model.summary()))
    print("predicting ...")
    y_pred = model.predict(x_test, batch_size=batch_size)
    score = model.evaluate(x_test, y_test, verbose=0)
    print(score)


num_classes = 10
batch_size = 64
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255
y_train = utils.to_categorical(y_train, num_classes)

x_test = x_test / 255
y_test = utils.to_categorical(y_test, num_classes)


input = Input(shape=(32, 32, 3), name='cifar_input')
vgg = vgg16.VGG16(include_top=False, input_tensor=input)

for layer in vgg.layers:
    layer.trainable = False

vgg.summary()

#Add new layers (fully-connected)
new_model = Sequential()
new_model.add(Flatten(input_shape=vgg.output_shape[1:], name='flatten'))
new_model.add(Dense(1024, activation='relu', name='fc1'))
new_model.add(Dropout(0.5))
new_model.add(Dense(10, activation='softmax', name='cifar_10_softmax'))

model = Model(inputs=vgg.input, outputs=new_model(vgg.output))
model.summary()

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

x_test = x_test / 255.
y_test = utils.to_categorical(y_test, num_classes)

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model_checkpoint = ModelCheckpoint(filepath='vgg_cifar10', verbose=1, save_best_only=True)

start = time.time()
model.fit(x_train, y_train, validation_data=(x_test[:5000], y_test[:5000]),
          callbacks=[model_checkpoint, early_stopping], epochs=100, batch_size=batch_size, verbose=2)
end = time.time()
print("Model took %0.2f seconds to train"%(end - start))

# Re-instantiate model to the best model saved
model = models.load_model('vgg_cifar10')

y_pred = model.predict(x_test, batch_size=batch_size)
score = model.evaluate(x_test, y_test, verbose=0)

print(score)
