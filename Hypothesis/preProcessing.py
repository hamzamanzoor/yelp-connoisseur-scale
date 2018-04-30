#import all necessary file handling libraries
import os
import gc
import sys

#import all datahandling and processing libraries
import pandas as pd
import numpy as np
import scipy as sp

#utility libraries
import pickle
import pprint as pp
import time
from random import shuffle


#load data and apply a header
dataDir = "dataset/" #replace as necessary
bus_revs = pd.read_csv(dataDir+"business_review.csv")   #replace filename with one exported from the SQL db.
bus_revs = bus_revs.rename(columns={'business_id': 'bus_id', 'review_rating': 'rating'})
np_bus_revs = bus_revs.as_matrix() #move from dataframe to numpy matrix for vectorized processing (avoid for loops).


users = pd.read_csv(dataDir+"user_frac.csv")
#use frequency to pre-calculate the bucket
# 0-10% --> bucket 0
#10-20% --> bucket 1
#20-30% --> bucket 2
#30-40% --> bucket 3
#40-50% --> bucket 4
#50-60% --> bucket 5
#60-70% --> bucket 6
#70-80% --> bucket 7
#80-90% --> bucket 8
#90-100%--> bucket 9
users['bucket'] = users['frequency'].apply(lambda x: int(np.floor(x*10))) 
np_users = users.as_matrix() #move from dataframe to numpy matrix for vectorized processing (avoid for loops).


# grab the first 1000 unique businesses after shuffle 
all_bus = bus_revs['bus_id'].unique()
shuffle(all_bus)
all_bus = all_bus[0:1000]


np_small_bus_revs = np_bus_revs[0, :]
i=0
for bus in all_bus:
    if (i%10 == 0):
        print i
    tiny_bus = np_bus_revs[(np_bus_revs[:, 1] == bus), :]
    np_small_bus_revs = np.vstack((np_small_bus_revs, tiny_bus))
    i = i+1


np_small_bus_revs = small_bus_revs[1:,:]
small_bus_revs = pd.DataFrame(np_small_bus_revs)
small_bus_revs.columns = ['user_id', 'bus_id', 'category', 'rating']
all_users = small_bus_revs['user_id'].unique()


# build a bucketloop up dictionary because it takes too long to search for user the users dataframe loaded earlier
bucketLookup = {}   # 2 level dctionary of form dict[user][] = [frequency, category_count, total_count, bucket] <-- this is a list
print "Number of users:\t", len(all_users)
i=0
for user in all_users:
    i = i+1
    if (i%100 == 0):
        print i
    tiny_user = np_users[(np_users[:, 0] == user), :]
    bucketLookup[user] = {}
    for j in range(tiny_user.shape[0]):
        bucketLookup[user][tiny_user[j,1]] = [int(tiny_user[j,2]), int(tiny_user[j,3]), int(tiny_user[j,4]), int(tiny_user[j,5])]
    


# save results as pickles to avoid doing processing again.
file_Name = "small_bus_revs.pkl"
fileObject = open(file_Name, 'wb')
pickle.dump(small_bus_revs, fileObject)
fileObject.close()

file_Name = "bucketLookup.pkl"
fileObject = open(file_Name, 'wb')
pickle.dump(bucketLookup, fileObject)
fileObject.close()