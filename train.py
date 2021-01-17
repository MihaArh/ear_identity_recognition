import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.vgg16 import preprocess_input

NCLASSES = 100
HEIGHT = 224
WIDTH = 224
NUM_CHANNELS = 3
BATCH_SIZE = 16

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])
rescale = layers.experimental.preprocessing.Rescaling(1. / 225)
base_model = tf.keras.applications.vgg16.VGG16(input_shape=None, include_top=False,
                                               weights='imagenet', classes=100)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)

output = layers.Dense(101, activation='softmax')(x)
# x = base_model.output
# x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dense(4096, activation='relu')(x)
# x = layers.Dense(1)(x)

model_3 = models.Model(inputs, output)
print(model_3.summary())

AUTOTUNE = tf.data.AUTOTUNE

train_dir = "data/train"
test_dir = "data/test"
train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=(HEIGHT, WIDTH))

val_batches = tf.data.experimental.cardinality(train_dataset)
validation_dataset = train_dataset.take(val_batches // 5)
train_dataset = train_dataset.skip(val_batches // 5)

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)

model_3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model_3.fit(train_dataset, epochs=30, validation_data=validation_dataset)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

model_3.save(f"models/model.h5")
