
from __future__ import print_function

#import all necessary file handling libraries
import os
import sys
from time import time
os.system('cls')

#import all datahandling and processing libraries
import numpy as np
import pandas as pd

# for plotting
import matplotlib
import matplotlib.pyplot as plt

#necessary keras stuff
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Input, GlobalMaxPooling1D, LSTM, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras import optimizers

#using some stuff from sklearn for keras
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import itertools
from pprint import pprint as ppr


# imprort the data for the six most reviewed resturants.
data = pd.read_pickle("review_6.pkl")
del data["rescaled_rating"] # delete the rescaled rating calculated form SQL as it was wrong


# get all uniques business ids as a list and sort them
business = data["business_id"].unique()
business.sort()

# pre-calcualte the average categorical visit frequency for the resturants and store in dict
freq_splits = dict()
for bus in business:
    freqs = data[data["business_id"] == bus]["freq"]
    print(len(freqs))
    freq_splits[bus] = np.mean(freqs)
ppr(freq_splits)



# implement the formula as layeed out in the paper
countT = 0
countF = 0
wf=5
we=1.5
new_ratings = []
lol=[]
for i, r in data.iterrows():
    if (r["freq"] < 1.2*freq_splits[r["business_id"]]): # if non-frequenter
        if r['review_rating'] > 3: #if positive bring review rating down
            countT += 1
            new_ratings.append(r['review_rating'] * ( wf*(r["freq"]/r["max_freq"]) + 
                                                      we*(2*r["exp"]/r["max_exp"]) ) / (wf+we*2))
        elif r['review_rating'] < 3: #if negative boost review rating
            countT += 1
            new_ratings.append(r['review_rating'] * (1 - (( wf*(r["freq"]/r["max_freq"]) + 
                                                      we*(2*r["exp"]/r["max_exp"]) ) / (wf+we*2))))
        else: # if neutral leave as is
            countT += 1
            new_ratings.append(r['review_rating'])
    else: #leave frequenter ratings as it
        countF += 1
        new_ratings.append(r['review_rating'])
        

#store back in the dataframe as a new column and ceil all values to the next integer e.g. 4.1 becomes 5
data['rescaled_rating'] = new_ratings
data['rescaled_rating'] = data['rescaled_rating'].apply(lambda x: np.ceil(x))


# build dictionaries for user in rank based rescaling
# this will help avoid looking up in the dataframe which is slow
freq_ranks = dict() #dict of form bus_id -> user_id -> rank of user in business   
exp_ranks = dict() #dict of form bus_id -> user_id -> rank of user in business   

# loop over all all bussinesses and for each loop over each user and then calcualte the rank for each user.
uniqs = list(data["business_id"].unique())
uniqs.sort()
for bus_id in uniqs:
    bus_df = data[data["business_id"] == bus_id]

    freq_ranks[bus_id] = dict()
    exp_ranks[bus_id] = dict()
    users = bus_df["user_id"].values
    
    user_freqs = bus_df["freq"].values
    bus_freq_ranks = np.argsort(user_freqs)
    for i in range(len(bus_freq_ranks)):
        freq_ranks[bus_id][users[i]] = len(bus_freq_ranks)-bus_freq_ranks[i]+1
    
    user_exps = bus_df["exp"].values
    bus_exp_ranks = np.argsort(user_exps)
    for i in range(len(bus_exp_ranks)):
        exp_ranks[bus_id][users[i]] = len(bus_exp_ranks)-bus_exp_ranks[i]+1




# implementation of rank based rescaling
# this uses a freq_ranks and ep_ranks to speed up the processing.
we = 1.5
wf = 5
new_rating = []
lens = len(data)
k=0
for ind, row in data.iterrows():
    if(k%1000 == 0):
        print(k, lens)
    # caluclate the weghted average of the exp rank and freq rank.
    # rescale using the avg and then store back in new_rating
    avg_rank = we*exp_ranks[row["business_id"]][row["user_id"]] + wf*freq_ranks[row["business_id"]][row["user_id"]]
    avg_rank = avg_rank/(we+wf)
    rat = row["review_rating"]*(avg_rank)/len(data[data["business_id"] == row["business_id"]])
    new_rating.append(rat)
    k=k+1


# find the hotel names for use in plotting labels
data.head()
hotels = data["name"].unique()



# plot everything distribution of the score before and after the rescaling
mybins=[1,2,3,4,5,6]
n=3
plt.figure(figsize=(5*n,5))
plt.subplot(1,n,1)
plt.hist(data["review_rating"], normed=True, bins=mybins)
plt.xlabel("Review rating", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
plt.title("Original")
plt.ylim([0, 1])
plt.grid()
plt.show()


plt.subplot(1,n,2)
plt.hist(data["rescaled_rating"], normed=True, bins=mybins)
plt.xlabel("Review rating", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
plt.title("Rescaled")
plt.ylim([0, 1])
plt.grid()
plt.show()


plt.subplot(1,3,3)
plt.hist(data["rescaled_rating2"], normed=True, bins=mybins)
plt.xlabel("review rating", fontsize=20)
plt.ylabel("count", fontsize=20)
plt.title("Hamza")
plt.ylim([0, 1])
plt.grid()
plt.show()



# jamal --> selective penalization scheme
# hamza --> rank based scheme
# this section simply spits the population into two portions
# 1) frequenters 2) non-frequenetrs
# we're using binary indexing to avoid explicity looping over all examples
# to learn about binary indexing please read up on numpy.
bus_id = data["business_id"].unique()
bus_id.sort()

freq_texts = []
nonfreq_texts = []

freq_original_rat = []
nonfreq_original_rat = []

freq_rescaled_jamal = []
nonfreq_rescaled_jamal = []

freq_rescaled_hamza = []
nonreq_rescaled_hamza = []

for bus in bus_id:
    bus_df = data[data["business_id"] == bus]
    
    freqs = bus_df["freq"].values
    freq_mask = (freqs > 1.2*np.mean(freqs))
    nonfreq_mask = (freqs <= 1.2*np.mean(freqs))
    
    freq_texts.append(bus_df[freq_mask]["text"].values)
    nonfreq_texts.append(bus_df[nonfreq_mask]["text"].values)

    freq_original_rat.append(bus_df[freq_mask]["review_rating"].values)
    nonfreq_original_rat.append(bus_df[nonfreq_mask]["review_rating"].values)

    freq_rescaled_jamal.append(bus_df[freq_mask]["rescaled_rating"].values)
    nonfreq_rescaled_jamal.append(bus_df[nonfreq_mask]["rescaled_rating"].values)

    freq_rescaled_hamza.append(bus_df[freq_mask]["rescaled_rating2"].values)
    nonreq_rescaled_hamza.append(bus_df[nonfreq_mask]["rescaled_rating2"].values)
    
    print(len(freq_texts[-1]), len(nonfreq_texts[-1]))




# set up direcotries for the glove embeddings
BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
MAX_SEQUENCE_LENGTH = 80
MAX_NUM_WORDS = 35000
EMBEDDING_DIM = 200


# read the embeddings from the files
print("=================================================================")
print('Indexing word vectors.')
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.'+str(EMBEDDING_DIM)+'d.txt'), encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))


# read  the text corpus on which the tokenizer was trained and retrain the tokenizer
# so that the semantics match
# a pickle could've been used as an alternative
reviews = pd.read_csv("review_500k.csv")
texts = (reviews["text"][0:300000]).values
print("=================================================================")
print("Text -> Integer tensor")
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
int_texts = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


# tokenize the reviews from review_6.pkl from list of words to list of ints
# do this for reviews of each hotel and
# do this separately for frequenetrs and non frequentrs portions
int_freq_texts = []
int_nonfreq_texts = []

for i in range(len(freq_texts)):
    sequences = tokenizer.texts_to_sequences(freq_texts[i])
    int_texts = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    int_freq_texts.append(int_texts)
    
    sequences = tokenizer.texts_to_sequences(nonfreq_texts[i])
    int_texts = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    int_nonfreq_texts.append(int_texts)
    
print(len(int_freq_texts), len(int_nonfreq_texts))


# load the GloVe into an emebedding matric for use in the embedding layer
print("=================================================================")
print('Preparing embedding matrix.')
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# define the model strcuture and then instead of training load the weigths from the checkpoint file.
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
Embedding_layer = Embedding(num_words,  EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
embedded_sequences = Embedding_layer(sequence_input)

lstm_l1 = LSTM(80, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))(embedded_sequences)
lstm_l2 = LSTM(80, recurrent_dropout=0.2, return_sequences=True)(lstm_l1)
lstm_l3 = LSTM(80, recurrent_dropout=0.2)(lstm_l2)

dens1 = Dense(int(80), activation='relu')(lstm_l3)
dens2 = Dense(int(80), activation='relu')(dens1)
dens3 = Dense(int(160), activation='relu')(dens2)
dens4 = Dense(int(160), activation='relu')(dens3)

preds = Dense(5, activation='softmax')(dens4)
model = Model(sequence_input, preds)
model.load_weights("./checkpoints/3stackLSTM-4dense-1523909298/250000-14-0.68.hdf5") # load weights from a saved cehckpoint file


adam = optimizers.Adam(clipnorm=1.0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
model.summary()



# for the 6 hotels
# predict the rating for frequenetrs
# predict the rating for non-frequenters
freq_preds = []
nonfreq_preds = []
for i in range(len(freq_texts)):
    print("=================================")
    print("Processing Hotel:\t",i)
    print("Predicting for frequenters")
    freq_preds.append(model.predict(int_freq_texts[i]))
    print("Predicting for non-frequenters")
    nonfreq_preds.append(model.predict(int_nonfreq_texts[i]))



# for each of the 6 hotel
# calulate the accuracy of the predicted rating of frequenters and non-frequenters
# when compared to the original rating scheme, the selective penalization scheme and the rank based scheme
# # we're using accuracy_score from sklearn to do this 
freq_original_acc = [] 
nonfreq_original_acc = [] 

freq_jamal_acc = [] 
nonfreq_jamal_acc = [] 

freq_hamza_acc = [] 
nonfreq_hamza_acc = [] 

for i in range(len(freq_texts)):
    pred = []
    for pr in freq_preds[i]:
        pred.append(int(np.argmax(pr)+1))
    freq_original_acc.append(accuracy_score(freq_original_rat[i], pred))
    freq_jamal_acc.append(accuracy_score(freq_rescaled_jamal[i], pred))
    freq_hamza_acc.append(accuracy_score(freq_rescaled_hamza[i], pred))
    
for i in range(len(nonfreq_texts)):
    pred = []
    for pr in nonfreq_preds[i]:
        pred.append(int(np.argmax(pr)+1))
    nonfreq_original_acc.append(accuracy_score(nonfreq_original_rat[i], pred))
    nonfreq_jamal_acc.append(accuracy_score(nonfreq_rescaled_jamal[i], pred))
    nonfreq_hamza_acc.append(accuracy_score(nonreq_rescaled_hamza[i], pred))



# FIONALLLLLLLLLLLYYYYYY
# plot the results we have calculated. 
# ^_______________________^
ind = np.arange(6)  # the x locations for the groups
width = 0.35       # the width of the bars
n=2
fig = plt.figure(figsize=(10,10))
rects1 = plt.bar(ind, freq_original_acc, width, label="freq")
rects1 = plt.bar(ind+width, nonfreq_original_acc, width, label="nonfreq")
plt.title("Original Yelp Scale", fontsize=20)
plt.ylabel("Accuracy", fontsize=20)
plt.xticks(ind, list(hotels), fontsize=10, rotation=45,)
plt.grid()
plt.ylim([0, 1])
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=15)
plt.legend(loc="upper right", fontsize=20)
plt.show()


fig = plt.figure(figsize=(10,10))
rects1 = plt.bar(ind, freq_jamal_acc, width, label="freq")
rects1 = plt.bar(ind+width, nonfreq_jamal_acc, width, label="nonfreq")
plt.title("Selective Penalization Scale", fontsize=20)
plt.ylabel("Accuracy", fontsize=20)
plt.xticks(ind, list(hotels), fontsize=10, rotation=45)
plt.grid()
plt.ylim([0, 1])
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=15)
plt.legend(loc="upper right", fontsize=20)
plt.show()


fig = plt.figure(figsize=(10,10))
rects1 = plt.bar(ind, freq_hamza_acc, width, label="freq")
rects1 = plt.bar(ind+width, nonfreq_hamza_acc, width, label="nonfreq")
plt.title("Ranking based Rescaling Scale", fontsize=20)
plt.ylabel("Accuracy", fontsize=20)
plt.xticks(ind, list(hotels), fontsize=10, rotation=45)
plt.grid()
plt.ylim([0, 1])
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=15)
plt.legend(loc="upper right", fontsize=20)
plt.show()



# here's mario :D 
'''
──────────────███████──███████
──────────████▓▓▓▓▓▓████░░░░░██
────────██▓▓▓▓▓▓▓▓▓▓▓▓██░░░░░░██
──────██▓▓▓▓▓▓████████████░░░░██
────██▓▓▓▓▓▓████████████████░██
────██▓▓████░░░░░░░░░░░░██████
──████████░░░░░░██░░██░░██▓▓▓▓██
──██░░████░░░░░░██░░██░░██▓▓▓▓██
██░░░░██████░░░░░░░░░░░░░░██▓▓██
██░░░░░░██░░░░██░░░░░░░░░░██▓▓██
──██░░░░░░░░░███████░░░░██████
────████░░░░░░░███████████▓▓██
──────██████░░░░░░░░░░██▓▓▓▓██
────██▓▓▓▓██████████████▓▓██
──██▓▓▓▓▓▓▓▓████░░░░░░████
████▓▓▓▓▓▓▓▓██░░░░░░░░░░██
████▓▓▓▓▓▓▓▓██░░░░░░░░░░██
██████▓▓▓▓▓▓▓▓██░░░░░░████████
──██████▓▓▓▓▓▓████████████████
────██████████████████████▓▓▓▓██
──██▓▓▓▓████████████████▓▓▓▓▓▓██
████▓▓██████████████████▓▓▓▓▓▓██
██▓▓▓▓██████████████████▓▓▓▓▓▓██
██▓▓▓▓██████████──────██▓▓▓▓████
██▓▓▓▓████──────────────██████ 
──████
'''