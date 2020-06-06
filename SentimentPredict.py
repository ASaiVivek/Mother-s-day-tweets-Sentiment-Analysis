
from tensorflow.keras.models import model_from_json
import numpy as np
import random
import os
import json
import csv
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer,tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras import datasets, layers, models, optimizers

embedding_dim = 100
max_length = 20
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
test_portion=.3


json_file=open('model.json','r')
loaded_model_json=json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")

loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

print("successfully loaded model")

with open('tokenizer.json') as f:
    data=json.load(f)
    tokenizer=tokenizer_from_json(data)

print("loaded tokenizer data")

corpus = []
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
        corpus.append(list_item)


training_size=len(corpus)

sentences=[]
labels=[]
random.shuffle(corpus)
for x in range(len(corpus)):
    sentences.append(corpus[x][0])
    labels.append(corpus[x][1])


word_index=tokenizer.word_index
vocab_size=len(word_index)

sequences=tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

split = int(test_portion * training_size)

test_sequences = padded[0:split]
training_sequences = padded[split:training_size]
test_labels = labels[0:split]
training_labels = labels[split:training_size]

#train_x=np.asarray(training_sequences)
train_y=np.asarray(training_labels)
#test_x=np.asarray(test_sequences)
test_y=np.asarray(test_labels)

ys=tf.keras.utils.to_categorical(train_y,num_classes=3)
yts=tf.keras.utils.to_categorical(test_y,num_classes=3)


score = loaded_model.evaluate(test_sequences, yts, verbose=1)
print(score)
valid_sentence= "Happy Mothers Day to all you amazing Mums out there! Keep being awesome! It's probably not been your normal mothers day but a mothers love is one that can overcome even the toughest of challenges. Hope you've all… https://www. instagram.com/p/B-C-0nZlrVe/ ?igshid=1e6tgbc5wrbx3 …"
valid_sequence=tokenizer.texts_to_sequences(valid_sentence)
valid_pad= pad_sequences(valid_sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#padded_data=valid_pad.flatten()

#valid_x=np.asarray(valid_pad)
print(valid_sequence)
print(valid_pad)
#print(padded_data)

#pred=loaded_model.predict_classes([valid_x])
#pred=np.argmax(loaded_model.predict(valid_pad),axis=0)
pred=loaded_model.predict_classes([valid_pad])
if pred.any()>1:
    prediction=1
elif pred.all()==1:
    prediction=0
else:
    prediction=-1
#pred=loaded_model.predict_classes(valid_pad)
print("X=%s , Predicted=%s"%(valid_pad,prediction))

test_data = []
valid_tweets=[]
print_data=[]
with open("dataset/test.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        print_data.append(row[0])
        valid_tweets.append(row[1])

print(valid_tweets[0])
print(valid_tweets[4])
print(print_data[0])
print(print_data[4])

print_data=np.asarray(print_data)
tweet_sequences=tokenizer.texts_to_sequences(valid_tweets)
tweets_padded = pad_sequences(tweet_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#print(tweets_padded[1])
#print(tweets_padded[9])


def sentiment_predict(valid_pad):
    pred=loaded_model.predict_classes([valid_pad])
    if pred.any()>1:
        return 1
    elif pred.all()==1:
        return 0
    else:
        return -1
with open('submissions.csv','w') as f:
    thewriter=csv.writer(f,delimiter='"',lineterminator='\n')
    thewriter.writerow(['id','sentiment_class'])
    for i in tweets_padded:
        sentiment=sentiment_predict(i)
        thewriter.writerow("{},{}".format(str(print_data[i][0]),str(sentiment)))
