# Yelp Connoisseur Scale
Bias and preference adjusted reviews on Yelp. A new rating scale on Yelp which which takse into account the tastes and preferences of the reviewers.

## Package Requirements

1: Python 3.5 or above (Anaconda 4.x preffered)

2: Numpy (latest)

3: Scipy (latest)

4: Matlplotlib (latest)

5: mySQL

6: Scikit Learn

7: Keras GPU

8: Yelp Dataset: https://www.yelp.com/dataset

9: GloVe embeddings (Wikipedia 2014 + Gigaword 5 version): https://nlp.stanford.edu/projects/glove/

For GPU requirements please look at the official Nvidia Docuemntation to install CUDA, cuDNN and other dependencies:
https://developer.nvidia.com/cuda-toolkit


## Data Pre-processing

Step 1: Deploy the Stored Procedure Data_Pre_Processing/review_business_join_long.sql <br />
Step 2: Deploy the Stored Procedure Data_Pre_Processing/friend_user_join.sql <br />
Step 3: Execute the SQL statements from Data_Pre_Processing/data_preparation.sql in sequence <br />


## Hypothesis

Step 1: Generate the necessary thin slice from SQL portion of the code (Data Pre-processing direccotry)<br />
Step 2: Change paths in Hypothesis/preProcessing.py to use the file from SQL<br />
Step 3: Run Hypothesis/preProcessing.py (This is a one time run ot generate the necessary pickles) <br />
Step 4: Run Hypothesis/hypothesis.py<br />

## Model

Step 1: Prepare a large slice from the yelp dataset. It must have:<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1- reivew text        <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2- review rating      <br />

Step 2: run Model/train_from_scratch.py. Input arguments are:   <br />    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1- sampleSize: number of examples to train on <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2- testSplit: value must be in interval (0, 1) <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3- validSplit: value must be in interval (0, 1) <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4- iterations: number of Epochs to train for <br />

(optional) Step 3: if you need to train from an existing checkpoint, use Model/train_from_checkpoint.py. You'll have to set the path for the checkpoint file inside the source code. The file should be of type hdf5 <br />

Step 4 : run Model/rescale_and_accuracy.py to calculate the rescaled ratings and the accuracies of the predicted ratings on the new scales.
