import os
import json
import re
import csv

os.getcwd()
os.chdir('C:/Users/Caio Brighenti/Documents/repositories/fake-news')

fnn_path = 'FakeNewsNet/code/fakenewsnet_dataset'


with open('FakeNewsNet/dataset/fnn_data.tsv', 'wt') as f:
    tsv_writer = csv.writer(f, delimiter='\t')
    tsv_writer.writerow(['ID','label','title', 'text'])


for dirName, subdirList, fileList in os.walk(fnn_path):
    #print('Found directory: %s' % dirName)
    for fname in fileList:
        #print('\t%s' % fname)
        with open(dirName + "/" + fname, "r") as f:
            ## load json
            datastore = json.load(f)
            ## grab needed parts
            title = datastore["title"]
            #title = re.sub('[^A-Za-z0-9]+', ' ', title)
            text = datastore["text"]
            text = re.sub(r'\n',' ', text)
            #text = re.sub('[^A-Za-z0-9]+', ' ', text)
            ## grab article ID
            ID = dirName.split("\\")[-1]
            ## grab truth label
            label = dirName.split("\\")[2]
            ## append to CSV
            with open('FakeNewsNet/dataset/fnn_data.tsv', 'a') as f:
                tsv_writer = csv.writer(f, delimiter='\t')
                tsv_writer.writerow([ID,label,title.encode("utf-8"), text.encode("utf-8")])
