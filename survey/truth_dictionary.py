import os
import re
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.book import *

#os.getcwd()
#os.chdir('C:/Users/Caio Brighenti/Documents/repositories/fake-news')

f = open('survey/survey_words.csv', 'r')
dat = pd.read_csv(f)
f.close()

## manipulate dataframe into flanneted vectors
dat = dat.drop([1,])
a = np.array(dat)
untruth = a[1:, [1,3,5,7,9,11,13,15,17,19] ].flatten(order='F')
truth = a[1:, [0,2,4,6,8,10,12,14,16,18] ].flatten(order='F')

# strip and clean
## untruth
untruth = untruth[~pd.isnull(untruth)]
untruth = [re.sub('\d+', '', x) for x in untruth]
untruth = [re.sub(r"([.!?,``$'':/()<>])", "", x) for x in untruth]
untruth = [re.sub("\xe2\x80\x94", "", x) for x in untruth]
untruth = [re.sub(r"wouldn\'t", "wouldnt", x) for x in untruth]
un_tokens = word_tokenize(''.join(untruth))
un_fd = dict(FreqDist(un_tokens))
del un_fd["''"]
del un_fd["--"]
## truth
truth = truth[~pd.isnull(truth)]
truth = [re.sub('\d+', '', x) for x in truth]
truth = [re.sub(r"([.!?,``$'':/()<>])", "", x) for x in truth]
truth = [re.sub("\xe2\x80\x94", "", x) for x in truth]
truth = [re.sub(r"wouldn\'t", "wouldnt", x) for x in truth]
tokens = word_tokenize(''.join(truth))
t_fd = dict(FreqDist(tokens))
del t_fd["''"]
del t_fd["--"]

## write dictionaries
un_df = pd.DataFrame([un_fd]).T
t_df = pd.DataFrame([t_fd]).T

f = open('survey/untruth_dict.csv', "w")
f.write(un_df.to_csv())
f.close()

f = open('survey/truth_dict.csv', "w")
f.write(t_df.to_csv())
f.close()
