import tensorflow as tf
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import os
from keras.callbacks import ModelCheckpoint

def train():
    # model init
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128,320, 3))
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))  # 2 output classes for binary classification

    # compile model
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    # data loading and augmentation
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        './data',
        target_size=(128,320),
        batch_size=4,
        class_mode='sparse',  # use sparse categorical since labels are integers
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        './data',
        target_size=(128,320),
        batch_size=4,
        class_mode='sparse',
        subset='validation')

    # callback to save the best model-----后加
    checkpoint = ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        epochs=20,
        callbacks = [checkpoint]  # include the checkpoint callback
    )

if __name__ == '__main__':
    train()
