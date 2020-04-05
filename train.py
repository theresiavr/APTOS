from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import pickle
import argparse
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import ResNet50

ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epoch', required=True,
                help = 'training epoch')
ap.add_argument('-b', '--batch_size', default=1,
                help = 'training batch size')
ap.add_argument('-opt', '--optimizer', required=True,
                help = 'optimizer for compile model option: sgd, adam, rmsprop, adagrad')
ap.add_argument('-t', '--transfer', default=False,
                help = 'transfer learning pretrain model, option: vgg16, vgg19, resnet50')
ap.add_argument('-l', '--loss', required=True,
                help = 'training loss to compile option: categorical_crossentropy, binary_crossentropy')
ap.add_argument('-act', '--activation', required=True,
                help = 'training activation function to compile, option: tanh, relu, sigmoid')
args = ap.parse_args()

PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_0_dir = os.path.join(train_dir, '0')
train_1_dir = os.path.join(train_dir, '1')
train_2_dir = os.path.join(train_dir, '2')
train_3_dir = os.path.join(train_dir, '3')
train_4_dir = os.path.join(train_dir, '4')

validation_0_dir = os.path.join(validation_dir, '0')
validation_1_dir = os.path.join(validation_dir, '1')
validation_2_dir = os.path.join(validation_dir, '2')
validation_3_dir = os.path.join(validation_dir, '3')
validation_4_dir = os.path.join(validation_dir, '4')

num_0_tr = len(os.listdir(train_0_dir))
num_1_tr = len(os.listdir(train_1_dir))
num_2_tr = len(os.listdir(train_2_dir))
num_3_tr = len(os.listdir(train_3_dir))
num_4_tr = len(os.listdir(train_4_dir))

num_0_val = len(os.listdir(validation_0_dir))
num_1_val = len(os.listdir(validation_1_dir))
num_2_val = len(os.listdir(validation_2_dir))
num_3_val = len(os.listdir(validation_3_dir))
num_4_val = len(os.listdir(validation_4_dir))

total_train = num_0_tr + num_1_tr + num_2_tr + num_3_tr + num_4_tr
total_val = num_0_val + num_1_val + num_2_val + num_3_val + num_4_val

print('total training num_0 images:', num_0_tr)
print('total training num_1 images:', num_1_tr)
print('total training num_2 images:', num_2_tr)
print('total training num_3 images:', num_3_tr)
print('total training num_4 images:', num_4_tr)

print('total validation num_0 images:', num_0_val)
print('total validation num_1 images:', num_1_val)
print('total validation num_2 images:', num_2_val)
print('total validation num_3 images:', num_3_val)
print('total validation num_4 images:', num_4_val)

print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

BATCH_SIZE = int(args.batch_size)
EPOCHS = int(args.epoch)
IMG_HEIGHT = 100
IMG_WIDTH = 100

# Build train data generator
train_image_generator = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=5,
                    horizontal_flip=True,
                    zoom_range=0.1,
                    brightness_range=[0.3,1.0]
                    )

# Build validation data generator
validation_image_generator = ImageDataGenerator(rescale=1./255)


train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='categorical',
                                                     classes=['0', '1', '2', '3',' 4'])
                                               
val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical',
                                                              classes=['0', '1', '2', '3',' 4'])

model = None
history = None

# Transfer learning with VGG16
if args.transfer == 'vgg16':
    conv_base = VGG16(weights='imagenet',
                         include_top=False,
                         input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Freeze pretrain model
    conv_base.trainable = False
    model = Sequential([
        conv_base,
        Flatten(),
        Dense(128, activation=args.activation),
        Dense(5, activation='softmax')
    ])
    print('[INFO] Transfer learning using VGG16')

elif args.transfer == 'vgg19':
    conv_base = VGG19(weights='imagenet',
                         include_top=False,
                         input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Freeze pretrain model
    conv_base.trainable = False
    model = Sequential([
        conv_base,
        Flatten(),
        Dense(128, activation=args.activation),
        Dense(5, activation='softmax')
    ])
    print('[INFO] Transfer learning using VGG19')

elif args.transfer == 'resnet50':
    conv_base = ResNet50(weights='imagenet',
                         include_top=False,
                         input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Freeze pretrain model
    conv_base.trainable = False
    model = Sequential([
        conv_base,
        Flatten(),
        Dense(128, activation=args.activation),
        Dense(5, activation='softmax')
    ])
    print('[INFO] Transfer learning using ResNet50')

else:
    model = Sequential([
        Conv2D(64, 3, padding='same', activation=args.activation,
                input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation=args.activation),
        Conv2D(32, 3, padding='same', activation=args.activation),
        MaxPooling2D(),
        Conv2D(16, 3, padding='same', activation=args.activation),
        Conv2D(16, 3, padding='same', activation=args.activation),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(5, activation='softmax')
        ])
    print('[INFO] Using simple architecture')

model.compile(optimizer=args.optimizer,
              loss=args.loss,
              metrics=["acc"])

# print model architecture
model.summary()

# Train model
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=total_val // BATCH_SIZE
)
history = history.history

# Save model history
with open('history_aptos', 'wb') as f:
    pickle.dump(history, f)

# Save model weights
model.save('aptos.h5')

acc = history['acc']
val_acc = history['val_acc']

loss = history['loss']
val_loss = history['val_loss']

epochs_range = range(EPOCHS)

# plot model accuracy and loss
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Save plot
plt.savefig('train_val.png')
