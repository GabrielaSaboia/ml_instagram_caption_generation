from keras_preprocessing.sequence import pad_sequences# from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, \
    TensorBoard
import time
import numpy as np
from PIL import Image
import os
import tensorflow as tf
import string
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


def extract_features(df):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    print(model.summary())
    features = {}
    images_list = df['image']
    img_path = 'backend/data/instadata/instagram_data'
    for i, name in tqdm(enumerate(images_list)):
        filename = os.path.join(os.getcwd(), img_path, name + '.jpg')
        image = load_img(filename,target_size=(224,224))
        image = img_to_array(image)
        image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image,verbose=0)
        image_id = name.split('.')[0]
        features[image_id] = feature
    return features



def preprocessing(dtexts, dimages):
    N = len(dtexts)
    print("# captions/images = {}".format(N))

    assert (N == len(dimages))

    Xtext, Ximage, ytext = [], [], []
    for text, image in zip(dtexts, dimages):
        for t in range(1, len(text)):
            in_text, out_text = text[:t], text[t]
            in_text = pad_sequences([in_text], maxlen=240).flatten()
            out_text = to_categorical(out_text, num_classes=5000)

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
    input_txt = layers.Input(shape=(240,))
    ftxt = layers.Embedding(5000, embedding_dimension,
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
    for iword in range(240):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], 240)
        yhat = img_text_model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        new_word = index_word[yhat]
        in_text += ' ' + new_word
        if new_word == '<END>':
            break
    return in_text

def train_model():


    df = process_data('/Users/neb/Desktop/hackathon/ml_instagram_caption_generation/backend/data/instadata/instagram_data/captions_csv.csv', '/Users/neb/Desktop/hackathon/ml_instagram_caption_generation/backend/data/instadata/instagram_data/captions_csv2.csv')

    images = extract_features(df)
    prop_test, prop_val = 0.2, 0.2

    N = df.shape[0]
    Ntest, Nval = int(N * prop_test), int(N * prop_val)

    def split_test_val_train(data_list, Ntest, Nval):
        return (data_list[:Ntest],
                data_list[Ntest:Ntest + Nval],
                data_list[Ntest + Nval:])

    dt_test, dt_val, dt_train = split_test_val_train(df['caption'], Ntest, Nval)
    images = list(images.items())
    images = np.asarray(images)

    di_test, di_val, di_train = split_test_val_train(images[:, 1], Ntest,
                                               Nval)
    fnm_test, fnm_val, fnm_train = split_test_val_train(images[:, 0], Ntest,
                                                        Nval)
    nb_words = 5000
    tokenizer = Tokenizer(num_words=nb_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['caption'][Ntest:])
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

    es = EarlyStopping(monitor='val_loss', patience=3,
                       restore_best_weights=True)

    hist = img_text_model.fit(train_dataset,
                              epochs=20, verbose=2,
                              batch_size=32,
                              validation_data=val_dataset,
                              callbacks=[es])  # pass callback to

    end = time.time()

    del Ximage_train
    del Xtext_train
    del ytext_train
    del Ximage_val
    del Xtext_val
    del ytext_val

    index_word = dict([(index, word) for word, index in tokenizer.word_index])
    Xtext_test, Ximage_test, ytext_test = preprocessing(dt_test, di_test)
    img_text_model.save('output/models/small_model.h5')

if __name__ == '__main__':
    train_model()




