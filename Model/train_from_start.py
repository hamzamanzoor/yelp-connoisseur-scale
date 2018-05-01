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
import itertools

#=====================================================================================
# function for plotting confusion matrix adapted from http://scikit-learn.org/stable/_downloads/plot_confusion_matrix.py
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')


        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        #plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label', fontsize=30)
        plt.xlabel('Predicted label', fontsize=30)




#=====================================================================================
sampleSize =  int(sys.argv[1]) # number of examples to trian on
testSplit = float(sys.argv[2]) # value must in (0, 1)
validSplit = float(sys.argv[3]) #vaue must in (0, 1)
iterations = int(sys.argv[4]) # number of Epochs to train for

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')  
MAX_SEQUENCE_LENGTH = 80
MAX_NUM_WORDS = 35000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = validSplit
TEST_SPLIT = testSplit
EPOCHS = iterations

#=================================================================
# first, build index mapping words in the embeddings set
# to their embedding vector
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


#=================================================================
# second, prepare text samples and their labels
print("=================================================================")
print('Processing text dataset')

reviews = pd.read_csv("review_500k.csv")  #replace filename as needed
texts = (reviews["text"][0:sampleSize]).values

labels = (reviews["review_rating"][0:sampleSize]).values
for i in range(len(labels)):
    labels[i] = labels[i]-1

unique_labels = list(np.unique(labels))
unique_labels.sort()

#dicionary built for ease of use
labels_index = dict()
labels_index['1'] = 0
labels_index['2'] = 1
labels_index['3'] = 2
labels_index['4'] = 3
labels_index['5'] = 4
print(labels_index)

#genrate class weights becuae the dataset is imbalanced and hence we need to penalise hte gradients of some classes.
#for details please read up on unblanced dataset training in neural networks
class_weights = class_weight.compute_class_weight('balanced', unique_labels, labels)
weight_dict = dict()
for i in range(len(class_weights)):
    weight_dict[i] = class_weights[i]
print('Found\t%s\ttext examples.' % len(texts))
print('Found\t%s\ttext labels.' % len(labels))
print('Class weights:\t', weight_dict)

#split up all senteces in the dataset and count the lenght i.e. number of words of each example.

review_lens = []
for i in range(len(texts)):
    review_lens.append(len(texts[i].split(" ")))

AVG_SEQUENCE_LENGTH = int(np.mean(review_lens))


# plot distribution of the review ratings
plt.figure(figsize = (20,9))
n, bins, patches = plt.hist(labels, bins=[0, 1, 2, 3, 4, 5, 6], normed=0, alpha=0.75)
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Distribution of rating scores')
plt.grid(True)
plt.show()

#plot distribution of the length of reviews.
plt.figure(figsize = (20,9))
n, bins, patches = plt.hist(review_lens, bins=list(np.arange(0,300,10)), normed=0, alpha=0.75)
plt.xlabel('review length')
plt.ylabel('Count')
plt.title('Distribution of rating scores')
plt.grid(True)
plt.xticks(np.arange(0,300,10))
plt.show()

#=================================================================
# finally, vectorize the text samples into a 2D integer tensor (list fo words -> list of integers)
# here we tokenize and filter all special symbols (it is done by default, please read: https://keras.io/preprocessing/text/#tokenizer)
# we also extend or truncate all reviews to MAX_SEQUENCE_LENGTH
print("=================================================================")
print("Text -> Integer tensor")
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


#=================================================================
# split the data into training, test and validation sets
# user random seeds as you see fit
print("=================================================================")
x_training, x_test, y_training, y_test = train_test_split(data, labels, test_size=TEST_SPLIT, random_state=1226)
x_train, x_val, y_train, y_val = train_test_split(x_training, y_training, test_size=VALIDATION_SPLIT, random_state=1993)

print("x_train:\t", x_train.shape)
print("x_val:\t", x_val.shape)
print("x_test:\t", x_test.shape)
print("y_train:\t", y_train.shape)
print("y_val:\t", y_val.shape)
print("x_test:\t", x_test.shape)


#=================================================================
# prepare embedding matrix
# this if for use in the embedding layer of the network
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


#=================================================================
# define the model and compile it
# structure is obvious form the code 
# NOTE: LSTM layers are NOT sequentially connected they are STACKED, this is obvios from the return_sequences=True argument
# this will also automcatically save checkpoints based on incresase in validation set accuracy.
# tensorboard logs will also be available in real time.
# the model uses adam with default settings (please read: https://keras.io/optimizers/#adam)
# we've added gradient clipping on the L2 norm of the gradient to avoid gradient explosion.
# depending on the machine a single epoch may take from 5 minutes upto a ~3 hours for 250,000 examples
("=================================================================")
print('Training model.')
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

preds = Dense(len(labels_index), activation='softmax')(dens4)
model = Model(sequence_input, preds)

adam = optimizers.Adam(clipnorm=1.0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
model.summary()

callbacks_list = []

modelDir ="./checkpoints/3stackLSTM-4dense-"+str(int(time()))+"/" 
os.mkdir(modelDir)
checkpath="{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(modelDir+str(sampleSize)+"-"+checkpath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


tensbdDir = "./logs/3stackLSTM-4dense-"+str(int(time()))+"/"
os.mkdir(tensbdDir)
tensorboard = TensorBoard(log_dir=tensbdDir)


callbacks_list = [tensorboard, checkpoint]
history = model.fit(x_train, y_train,  batch_size=256, epochs=EPOCHS, validation_data=(x_val, y_val), class_weight=weight_dict, callbacks=callbacks_list)

#==============================================================================================
# plot the model performance over time (per epoch)
print("=================================================================")
fig = plt.figure(figsize =(20,9))
ax = fig.gca()
plt.grid()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('Accuracy', fontsize = 30)
plt.xlabel('Epoch', fontsize = 30)
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left', fontsize = 30)
plt.savefig("our_acc.png")

fig = plt.figure(figsize =(20,9))
ax = fig.gca()
plt.grid()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss', fontsize = 30)
plt.xlabel('Epoch', fontsize = 30)
plt.legend(['Training Loss', 'Validation loss'], loc='upper left', fontsize = 30)


#==============================================================================================
# This portion generates and plots confusion matricies
predictions = model.predict(x_test)
y_pred = []
y_base = []
for pred in predictions:
    y_pred.append(np.argmax(pred))
for base in y_test:
    y_base.append(np.argmax(base))
    
cnf_matrix_test = confusion_matrix(y_base, y_pred)

fig = plt.figure(figsize =(20,20))
plot_confusion_matrix(cnf_matrix_test, classes=["1","2", "3", "4", "5"], normalize=True, title='Normalized confusion matrix (test set)')
plt.show()


predictions = model.predict(x_training)
y_pred = []
y_base = []
for pred in predictions:
    y_pred.append(np.argmax(pred))
for base in y_training:
    y_base.append(np.argmax(base))
    
cnf_matrix_train = confusion_matrix(y_base, y_pred)

fig = plt.figure(figsize =(20,20))
plot_confusion_matrix(cnf_matrix_train, classes=["1","2", "3", "4", "5"], normalize=True, title='Normalized confusion matrix (train set)')
plt.show()