import tensorflow as tf
import csv
import random
import numpy as np
import json

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras import datasets, layers, models, optimizers

vocab_size=1000
embedding_dim = 100
max_length = 20
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
test_portion=.3

corpus = []
num_sentences = 0

class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy')>0.99):
                self.model.stop_training = True

callbacks = myCallback()

with open("dataset/train.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        list_item=[]
        list_item.append(row[1])
        this_label=row[5]
        if this_label=='0':
            list_item.append(1)
        elif this_label=='-1':
            list_item.append(0)
        else:
            list_item.append(2)
        num_sentences = num_sentences + 1
        corpus.append(list_item)

print(num_sentences)
print(len(corpus))
print(corpus[3])

training_size=len(corpus)

sentences=[]
labels=[]
random.shuffle(corpus)
for x in range(len(corpus)):
    sentences.append(corpus[x][0])
    labels.append(corpus[x][1])

tokenizer=Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(sentences)
word_index=tokenizer.word_index
vocab_size=len(word_index)

sequences=tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

split = int(test_portion * training_size)

test_sequences = padded[0:split]
training_sequences = padded[split:training_size]
test_labels = labels[0:split]
training_labels = labels[split:training_size]

train_x=np.asarray(training_sequences)
train_y=np.asarray(training_labels)
test_x=np.asarray(test_sequences)
test_y=np.asarray(test_labels)
#print(train_x)
#print(train_y)
ys=tf.keras.utils.to_categorical(training_labels,num_classes=3)
yts=tf.keras.utils.to_categorical(test_labels,num_classes=3)
print(vocab_size)
print(word_index['happy'])
#print(ys)
print(training_sequences[3])
#print(ys[3])
print(ys[3])
#print(train_y)
model = tf.keras.Sequential([
 tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
 #tf.keras.layers.Conv1D(64, 5, activation='relu'),
 #tf.keras.layers.MaxPooling1D(pool_size=4),
 #tf.keras.layers.LSTM(64)
 #tf.keras.layers.Flatten(),
 tf.keras.layers.GlobalAveragePooling1D(),
 tf.keras.layers.Dense(24, activation='relu'),
 #tf.keras.layers.Dense(9,activation='relu'),
 tf.keras.layers.Dense(3, activation='softmax')
 ])

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 100

history = model.fit(training_sequences, ys , epochs=num_epochs, validation_data=(test_sequences,yts), verbose=1,callbacks=[callbacks])

model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
print("updated model")
model.save_weights("model.h5")
print('updated weights')
tokenizer_json=tokenizer.to_json()
with open('tokenizer.json','w',encoding='utf-8')as f:
    f.write(json.dumps(tokenizer_json,ensure_ascii=False))
print("updated tokenizer")
#valid_sentence="Happy mother's day stay safe " #though it's a western culture i don't celebrate "
#valid_sequence=tokenizer.texts_to_sequences(valid_sentence)
#valid_pad= pad_sequences(valid_sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
#valid_x=np.asarray(valid_pad)

#print(model.predict_classes([valid_x]))
#print(np.argmax(model.predict(valid_x), axis=-1))
#do one hot encoding course 3 week 4 video more on training the data nlp course

#predict_classes(valid_x)...check image taken in phone for reference
#check input shape in embeddings for better accuracy
