from keras_preprocessing.sequence import pad_sequences# from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, \
    TensorBoard
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import tensorflow as tf
import string
import pickle as pkl
import psutil
from pickle import dump
from pickle import load
from keras.applications.xception import Xception #to get pre-trained model Xception
from keras.applications.xception import preprocess_input
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras_preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Input, Dense#Keras to build our CNN and LSTM
from keras.layers import LSTM, Embedding, Dropout
from keras.models import Model
from os import listdir
from keras.applications.vgg16 import VGG16, preprocess_input
from tqdm import tqdm
 #to check loop progress
from keras import layers
from data_munger import process_data
MAXLEN = 500
VOCAB_SIZE = 50
OUTPUT_PATH = 'C:\\Users\\nebul\Desktop\\instahack\\ml_instagram_caption_generation\\backend\\output'
def extract_features(df):
    model = VGG16(include_top=True, weights='imagenet')
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    print("checking gpu")
    print(tf.test.is_gpu_available())
    print(model.summary())
    features = {}
    images_list = df['Image File']
    img_path = 'backend/data/instadata/instagram_data'
    for i, name in tqdm(enumerate(images_list)):
        filename = os.path.join(os.getcwd(), img_path, f"{name}.jpg")
        image = load_img(filename,target_size=(224,224,3))
        image = img_to_array(image)
        # image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image.reshape((1,) + image.shape[:3]),verbose=0)
        image_id = name.split('.')[0]
        features[image_id] = feature.flatten()
    return features



def preprocessing(dtexts, dimages):
    N = len(dtexts)
    print("# captions/images = {}".format(N))

    assert (N == len(dimages))

    Xtext, Ximage, ytext = [], [], []
    for text, image in zip(dtexts, dimages):
        for t in range(1, len(text)):
            in_text, out_text = text[:t], text[t]
            in_text = pad_sequences([in_text], maxlen=MAXLEN).flatten()
            out_text = to_categorical(out_text, num_classes=VOCAB_SIZE)

            Xtext.append(in_text)
            Ximage.append(image)
            ytext.append(out_text)

    Xtext = np.array(Xtext)
    Ximage = np.array(Ximage)
    ytext = np.array(ytext)
    print("Xtext: {}, Ximage: {}, ytext: {}".format(Xtext.shape, Ximage.shape,
                                                    ytext.shape))
    return (Xtext, Ximage, ytext)


def img_to_text_model(input_shape):
    embedding_dimension = 64
    input_image = layers.Input(shape=(input_shape,))
    fimage = layers.Dense(256, activation='relu', name="ImageFeature")(
        input_image)
    input_txt = layers.Input(shape=(MAXLEN,))
    ftxt = layers.Embedding(VOCAB_SIZE, embedding_dimension,
                            mask_zero=True)(input_txt)
    ftxt = layers.LSTM(256, name="CaptionFeature")(ftxt)
    decoder = layers.add([ftxt, fimage])
    decoder = layers.Dense(256, activation='relu')(decoder)
    output = layers.Dense(5000, activation='softmax')(decoder)
    model = Model(inputs=[input_image, input_txt], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model




def predict_caption(img_text_model, image, tokenizer, index_word):
    in_text = '<START>'
    for iword in range(MAXLEN):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], MAXLEN)
        yhat = img_text_model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        new_word = index_word[yhat]
        in_text += ' ' + new_word
        if new_word == '<END>':
            break
    return in_text

def metric(hist):
    for label in ['loss', 'val_loss']:
        plt.plot(hist.history[label], label=label)
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    plt.savefig(os.path.join(OUTPUT_PATH, 'metrics.png'))


def plot_images(img_text_model, fnm_test, di_test, tokenizer, index_word):
    npic = 5
    npix = 224
    target_size = (npix, npix, 3)
    count = 1
    fig = plt.figure(figsize=(10, 20))
    for jpgfnm, image_feature in zip(fnm_test[:npic], di_test[:npic]):
        # images
        filename = os.path.join('data/instadata/instagram_data', jpgfnm)
        image_load = load_img(filename, target_size=target_size)
        ax = fig.add_subplot(npic, 2, count, xticks=[], yticks=[])
        ax.imshow(image_load)
        count += 1

        # caption
        caption = predict_caption(img_text_model,
                                  image_feature.reshape(1, len(image_feature)),
                                  tokenizer,
                                  index_word)
        ax = fig.add_subplot(npic, 2, count)
        plt.axis('off')
        ax.plot()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0, 0.5, caption, fontsize=20)
        count += 1
    print('[INFO] displaying sample results')
    plt.show()
    plt.savefig(os.path.join(OUTPUT_PATH, 'sample_output.png'))


def train_model():

    firstcsv = r"C:\\Users\\nebul\Desktop\\instahack\\ml_instagram_caption_generation\backend\data\\instadata\instagram_data\\captions_csv.csv"
    secondcsv = r"C:\\Users\\nebul\Desktop\\instahack\\ml_instagram_caption_generation\backend\data\\instadata\instagram_data\\captions_csv2.csv"

    df = process_data(firstcsv)
    images = []
    fname = f"{OUTPUT_PATH}\\small_image_features_dictionary.pkl"
    try:
        images = pkl.load(open(fname, 'rb'))
    except:
        images = extract_features(df)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    pkl.dump(images, open(fname, 'wb'),
              protocol=pkl.HIGHEST_PROTOCOL)
    prop_test, prop_val = 0.1, 0.2

    N = df.shape[0]
    Ntest, Nval = int(N * prop_test), int(N * prop_val)

    def split_test_val_train(data_list, Ntest, Nval):
        return (data_list[:Ntest],
                data_list[Ntest:Ntest + Nval],
                data_list[Ntest + Nval:])

    dt_test, dt_val, dt_train = split_test_val_train(df['Caption'], Ntest, Nval)
    images = list(images.items())
    images = np.asarray(images)

    di_test, di_val, di_train = split_test_val_train(images[:, 1], Ntest,
                                               Nval)
    fnm_test, fnm_val, fnm_train = split_test_val_train(images[:, 0], Ntest,
                                                        Nval)
    global MAXLEN
    MAXLEN = np.max([len(text) for text in df['Caption']])
    nb_words = 5000
    tokenizer = Tokenizer(num_words=nb_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['Caption'][Ntest:])
    global VOCAB_SIZE
    VOCAB_SIZE = len(tokenizer.word_index) + 1
    dt_train = tokenizer.texts_to_sequences(dt_train)
    dt_test = tokenizer.texts_to_sequences(dt_test)
    dt_val = tokenizer.texts_to_sequences(dt_val)
    Xtext_train, Ximage_train, ytext_train = preprocessing(dt_train,
                                                           di_train)

    del dt_train
    del di_train
    del dt_val
    del di_val
    del df
    del images

    train_input_set = tf.data.Dataset.from_tensor_slices((Ximage_train,
                                                          Xtext_train))
    train_label_set = tf.data.Dataset.from_tensor_slices(ytext_train)
    train_dataset = tf.data.Dataset.zip((train_input_set, train_label_set))
    train_dataset = train_dataset.shuffle(100).batch(
        32)  # .prefetch(buffer_size=2)
    val_input_set = tf.data.Dataset.from_tensor_slices((Ximage_val, Xtext_val))
    val_label_set = tf.data.Dataset.from_tensor_slices(ytext_val)
    val_dataset = tf.data.Dataset.zip((val_input_set, val_label_set))
    val_dataset = val_dataset.batch(32)

    img_text_model = img_to_text_model(Ximage_train.shape[1])

    start = time.time()

    checkpoint_path = os.path.join(OUTPUT_PATH, "checkpoints",
                                   'cp-{epoch:02d}-{val_loss:.2f}.hdf5')
    checkpoint_dir = os.path.dirname(checkpoint_path)

    es = EarlyStopping(monitor='val_loss', patience=3,
                       restore_best_weights=True)
    mc = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',
                         save_best_only=False, save_weights_only=True)
    tb = TensorBoard(log_dir=f'{OUTPUT_PATH}\\logs', histogram_freq=1, write_graph=True,
                     embeddings_freq=1)

    hist = img_text_model.fit(train_dataset,
                              epochs=20, verbose=2,
                              batch_size=32,
                              validation_data=val_dataset,
                              callbacks=[es,mc,tb])  # pass callback to

    end = time.time()

    del Ximage_train
    del Xtext_train
    del ytext_train
    del Ximage_val
    del Xtext_val
    del ytext_val

    metric(hist)

    index_word = dict([(index, word) for word, index in tokenizer.word_index])
    plot_images(img_text_model, fnm_test, di_test, tokenizer, index_word)
    Xtext_test, Ximage_test, ytext_test = preprocessing(dt_test, di_test)
    fname = f'{OUTPUT_PATH}\\models\\small_model.h5'
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    img_text_model.save(fname)
def get_caption(image):
    nb_words = 5000
    tokenizer = Tokenizer(num_words=nb_words, oov_token='<OOV>')
    index_word = dict([(index, word) for word, index in tokenizer.word_index])
    fname = f'{OUTPUT_PATH}\\models\\small_model.h5'
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    modelCaption = load_model(fname)
    image_feature = None
    model = VGG16(include_top=True, weights='imagenet')
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    image = load_img(image,target_size=(224,224,3))
    image = img_to_array(image)
    image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    image = preprocess_input(image)
    image_feature = model.predict(image.reshape((1,) + image.shape[:3]),verbose=0).flatten()
    return predict_caption(modelCaption, image_feature, tokenizer,index_word )


if __name__ == '__main__':
    train_model()