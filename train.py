from tensorflow.keras import metrics
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
                help = 'training loss to compile, option: tanh, relu, sigmoid')
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

batch_size = int(args.batch_size)
epochs = int(args.epoch)
IMG_HEIGHT = 100
IMG_WIDTH = 100


train_image_generator = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=5,
                    horizontal_flip=True,
                    zoom_range=0.1,
                    brightness_range=[0.3,1.0]
                    )
                    
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
                    
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='categorical',
                                                     classes=['0', '1', '2', '3',' 4'])
                                               
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical',
                                                              classes=['0', '1', '2', '3',' 4'])
                                                              
sample_training_images, _ = next(train_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
#plotImages(sample_training_images[:5])

model = None
history = None

if args.transfer == 'vgg16':
    conv_base = VGG16(weights='imagenet',
                         include_top=False,
                         input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # for layer in conv_base.layers[-2]:
    #    layer.trainable=False

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

    # for layer in conv_base.layers[-2]:
    #    layer.trainable=False

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

    # for layer in conv_base.layers[-2]:
    #    layer.trainable=False

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
        #Dropout(0.2),
        Conv2D(32, 3, padding='same', activation=args.activation),
        Conv2D(32, 3, padding='same', activation=args.activation),
        MaxPooling2D(),
        Conv2D(16, 3, padding='same', activation=args.activation),
        Conv2D(16, 3, padding='same', activation=args.activation),
        MaxPooling2D(),
        #Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(5, activation='softmax')
        ])

    print('[INFO] Using simple architecture')
model.compile(optimizer=args.optimizer,
              loss=args.loss,
              metrics=["acc"])
              
model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)
history = history.history

with open('history_aptos', 'wb') as f:
    pickle.dump(history, f)

model.save('aptos.h5')

acc = history['acc']
val_acc = history['val_acc']

loss = history['loss']
val_loss = history['val_loss']

epochs_range = range(epochs)

print(model.inputs)
print(model.outputs)

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

import tensorflow as tf

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


frozen_graph = freeze_session(tf.compat.v1.keras.backend.get_session(),
                              output_names=[out.op.name for out in model.outputs])

tf.io.write_graph(frozen_graph, "model", "tf_model.pb", as_text=False)


plt.savefig('train_val.png')
