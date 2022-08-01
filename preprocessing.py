#read in Train_Tagged_Titiles.tsv and put into pandas dataframe
import pandas as pd
import csv
import numpy as np

df = pd.read_csv("Train_Tagged_Titles.tsv", sep="\t", skiprows=1, error_bad_lines=False, quoting=csv.QUOTE_NONE, header=None)
#print number of rows in df
print(df.head())
#print number of rows in df
print(df.shape)
#print df.head
print(df.head())
 #made header of each column in df id, title, entity, tag
df.columns = ['id', 'title', 'entity', 'tag']
#print df.head

print(df.head())

#if a row has tag of NaN, then add the entity string with the previous row's entity string

#print df.head  
#convert df to array of tuples
df_array = df.values
#loop through df_array backwards ignoring the first row
for i in range(len(df_array)-1, 0, -1):
    #if df_array[i][3] is NaN
    if pd.isnull(df_array[i][3]):
        #set df_array[i][3] to df_array[i-1][3]
        df_array[i-1][2] += " "+ df_array[i][2]
        #drop i row

        df_array = np.delete(df_array, i, 0)
#print df_array
#print df_array[0]
print(df_array[0])
#converrt df_array into df
train_data=[]
for entity in df_array:
    location=entity[1].find(entity[2])
    length=len(entity[2])
    train_data.append(
        
        (entity[1], [(location, location+length,entity[3])])
    )

print(train_data[0][0])
print(train_data[1][1][0])
train_data1=[]
lol = ''
currentEList = []
for i in range(1, len(train_data)):
    asdf = train_data[i][1][0]
    if train_data[i][0] == lol:
        currentEList.append((asdf[0], asdf[1], asdf[2]))

    # if train_data[i][0] == train_data[i+1][0]:
    #     train_data1.append(train_data[i][0], [(train_data[i][1][0][0], train_data[i][1][0][1], train_data[i][1][0][2])])
    else:
        train_data1.append((train_data[i-1][0], currentEList))
        currentEList = []
print(train_data1[0:50])
'''
for each row:
if string != last create new row
else add entity to tuple list'''



import spacy
from spacy.tokens import DocBin
nlp = spacy.blank("en")
db = DocBin()
for text, annotations in train_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations:
        span = doc.char_span(start, end, label=label)
        ents.append(span)
    print(doc.ents)
    doc.ents = ents
    db.add(doc)
db.to_disk("./train.spacy")

import pickle

import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding

# with open('train_data.spacy', 'wb') as f:
#     pickle.dump(train_data, f)





#print df.head 