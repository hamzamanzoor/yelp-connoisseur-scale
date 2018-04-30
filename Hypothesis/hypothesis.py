#import all necessary file handling libraries
import os
import gc
import sys

#import all datahandling and processing libraries
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats


#utility libraries
import pickle
import pprint as pp
import time
from random import shuffle

#for plotting
import matplotlib
import matplotlib.pyplot as plt

#not necessary to load ... you can skip  it
from IPython.display import display  
from ipywidgets import FloatProgress 



# load the dicionaries generated from preProcessing.py
dataDir = "dataset/"
with open("small_bus_revs.pkl",'r') as f:
    small_bus_revs = pickle.load(f)

print small_bus_revs.head()

with open("bucketLookup.pkl",'r') as f:
    bucketLookup = pickle.load(f)


#build dictionaries to avoid use of dataframes because data frames are slow for lookups
#these dictionaries are used in hypothesis.py to build the bar charts.
all_bus = small_bus_revs['bus_id'].unique()
scoresDict = dict()  #2 level dictionary of form dict[buss_ids][category] = review_rating
countsDict = dict()  #2 level dictionary of form dict[buss_ids][category] = review_count
fp = FloatProgress(min=0,max=len(all_bus)) 
display(fp)
errorcount = 0
for i in range(len(all_bus)):
    start_time = time.time()
    if (i%1)==0:
        string = str(i)+'/1000% \r'
        sys.stdout.write(string)
        sys.stdout.flush()
    bus_df = small_bus_revs[(small_bus_revs['bus_id'] == all_bus[i])]
    scoresDict[all_bus[i]] = {}
    countsDict[all_bus[i]] = {}
    all_cats = bus_df['category'].unique()
    for j in range(len(all_cats)):
        cat_df = bus_df[(bus_df['category'] == all_cats[j])]
        cat_users = cat_df['user_id'].unique()
        score = np.zeros(10)
        count = np.zeros(10)
        for user in cat_users:
            #bucket = int(users.loc[(users['user_id'] == user) & (users['category'] == all_cats[j])]['bucket'])
            try:
                bucket = bucketLookup[user][all_cats[j]][-1]
                if bucket == 10:
                    bucket = bucket - 1
                score[bucket] = score[bucket] + int(cat_df.loc[(cat_df['user_id'] == user) & (cat_df['category'] == all_cats[j])]['rating'])
                count[bucket] = count[bucket] + 1
            except:
                errorcount = errorcount + 1
        scoresDict[all_bus[i]][all_cats[j]] = score
        countsDict[all_bus[i]][all_cats[j]] = count
    fp.value += 1
    
print "\n\nError Count:\t",errorcount,"\n\n"



# save to pickle to avoid processing again.
file_Name = "scoresDict.pkl"
fileObject = open(file_Name, 'wb')
pickle.dump(scoresDict, fileObject)
fileObject.close()

file_Name = "countsDict.pkl"
fileObject = open(file_Name, 'wb')
pickle.dump(countsDict, fileObject)
fileObject.close()



# regarrange the data in scoresDict and countsDict into a 2D ndarrays for ease of use while plottig
scores = np.zeros((1,10))
for key in scoresDict.keys():
    for l2Key in scoresDict[key].keys():
        scores = np.vstack((scores, scoresDict[key][l2Key]))
scores = scores[1:, :]    

counts = np.zeros((1,10))
for key in countsDict.keys():
    for l2Key in countsDict[key].keys():
        counts = np.vstack((counts, countsDict[key][l2Key]))
counts = counts[1:, :]    


#aggregate the review ratings and the review counts per bucket
overallScores = np.sum(scores, axis=0)
overallCounts = np.sum(counts, axis=0)

ind = np.arange(10)  # the x locations for the groups
width = 0.45       # the width of the bars
fig, ax = plt.subplots(figsize = (20, 9))
p1 = ax.bar(ind-0.5*width, overallScores, width, label = "Aggregate Review Rating")
p2 = ax.bar(ind+0.5*width, overallCounts, width, label = "Aggregate Num of Reviews")
plt.ylabel('Aggregate value (Log Scale)', fontsize = 30)
plt.xlabel('Categorical visit frequency', fontsize = 30)
plt.xticks(ind, ('0%-10%', '10%-20%', '20%-30%','30%-40%', '40%-50%', '50%-60%','60%-70%', '70%-80%', '80%-90%', '90%-100%'), fontsize = 18)
plt.yticks(fontsize = 20)
plt.yscale('log')
plt.legend(fontsize = 30, loc = "best")
plt.show()

ind = np.arange(10)  # the x locations for the groups
width = 0.45       # the width of the bars
fig, ax = plt.subplots(figsize = (20, 9))
p1 = ax.bar(ind-0.5*width, overallScores, width, label = "Aggregate Review Rating")
p2 = ax.bar(ind+0.5*width, overallCounts, width, label = "Aggregate Num of Reviews")
plt.ylabel('Aggregate value', fontsize = 30)
plt.xlabel('Categorical visit frequency', fontsize = 30)
plt.xticks(ind, ('0%-10%', '10%-20%', '20%-30%','30%-40%', '40%-50%', '50%-60%','60%-70%', '70%-80%', '80%-90%', '90%-100%'), fontsize = 18)
plt.yticks(fontsize = 20)
plt.legend(fontsize = 30, loc = "best")
plt.show()


#=========================================================
# searate poppulation into two chunks
# buckets  0 and 1 i.e. 0-10% and 10-20% are being considered as non-frequnetrs.
freqs = []
nonFreqs = []

fp = FloatProgress(min=0,max=len(all_bus)) 
display(fp)
errorcount = 0

all_bus = small_bus_revs['bus_id'].unique()
for i in range(len(all_bus)):
    if (i%10)==0:
        string = str(i)+'/1000% \r'
        sys.stdout.write(string)
        sys.stdout.flush()
    bus_df = small_bus_revs[(small_bus_revs['bus_id'] == all_bus[i])]
    all_cats = bus_df['category'].unique()
    for j in range(len(all_cats)):
        cat_df = bus_df[(bus_df['category'] == all_cats[j])]
        cat_users = cat_df['user_id'].unique()
        for user in cat_users:
            try:
                bucket = bucketLookup[user][all_cats[j]][-1]
                if bucket == 0 or bucket == 1:
                    nonFreqs.append(int(cat_df.loc[(cat_df['user_id'] == user) & 
                                                (cat_df['category'] == all_cats[j])]['rating']))
                else:
                    freqs.append(int(cat_df.loc[(cat_df['user_id'] == user) & 
                                                (cat_df['category'] == all_cats[j])]['rating']))
                    
            except:
                errorcount = errorcount + 1
    fp.value += 1
    
print "\n\nError Count:\t",errorcount,"\n\n"

print "Test for Null-Hypothesis:"
print stats.ttest_ind(freqs, nonFreqs, equal_var = False)