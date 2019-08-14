import os
import json
import re
import csv
import pandas as pd


os.getcwd()
os.chdir('C:/Users/Caio Brighenti/Documents/repositories/fake-news')

fnn_path = 'FakeNewsNet/code/fakenewsnet_dataset'

## full data
with open('FakeNewsNet/dataset/fnn_data.tsv', 'wt', encoding = "utf-8") as f:
    tsv_writer = csv.writer(f, delimiter='\t')
    tsv_writer.writerow(['ID','label','title', 'text'])

## train set
with open('FakeNewsNet/dataset/fnn_train.tsv', 'wt', encoding = "utf-8") as f:
    tsv_writer = csv.writer(f, delimiter='\t')
    tsv_writer.writerow(['ID','label','title', 'text'])

## test set
with open('FakeNewsNet/dataset/fnn_test.tsv', 'wt', encoding = "utf-8") as f:
    tsv_writer = csv.writer(f, delimiter='\t')
    tsv_writer.writerow(['ID','label','title', 'text'])

count = 0
for dirName, subdirList, fileList in os.walk(fnn_path):
    for fname in fileList:
        count+=1
        with open(dirName + "/" + fname, "r", encoding ='utf-8') as f:
            ## load json
            datastore = json.load(f)
            ## grab needed parts
            title = datastore["title"]
            #title = re.sub('[^A-Za-z0-9\'"`.,?!]+', ' ', title)
            text = datastore["text"]
            text = re.sub(r'\n',' ', text)
            text = re.sub('\"\"\"','', text)
            #text = re.sub('[^A-Za-z0-9\'"`.,?!]+', ' ', text)
            ## grab article ID
            ID = dirName.split("\\")[-1]
            ## grab truth label
            label = dirName.split("\\")[2]
            ## append to CSV
            with open('FakeNewsNet/dataset/fnn_data.tsv', 'a', encoding='utf-8') as out:
                tsv_writer = csv.writer(out, delimiter='\t')
                tsv_writer.writerow([ID,label,title, text])
            ## 75/25 split
            if count % 4 == 0:
                path = 'FakeNewsNet/dataset/fnn_test.tsv'
            else:
                path = 'FakeNewsNet/dataset/fnn_train.tsv'
            ## write to train or test
            with open(path, 'a', encoding='utf-8') as file:
                tsv_writer = csv.writer(file, delimiter='\t')
                tsv_writer.writerow([ID,label,title, text])


## write csv versions
train_path='FakeNewsNet/dataset/fnn_train.tsv'
test_path='FakeNewsNet/dataset/fnn_test.tsv'
train_table=pd.read_table(train_path,sep='\t')
train_table.to_csv('FakeNewsNet/dataset/fnn_train.csv',index=False)
test_table=pd.read_table(test_path,sep='\t')
test_table.to_csv('FakeNewsNet/dataset/fnn_test.csv',index=False)
