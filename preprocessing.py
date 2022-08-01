#read in Train_Tagged_Titiles.tsv and put into pandas dataframe
import pandas as pd
import csv
import numpy as np

df = pd.read_csv("Train_Tagged_Titles.tsv", sep="\t", skiprows=1, error_bad_lines=False, quoting=csv.QUOTE_NONE, header=None)
 #made header of each column in df id, title, entity, tag
df.columns = ['id', 'title', 'entity', 'tag']
#convert df to array of tuples
df_array = df.values

print("Collapsing tags...")
#loop through df_array backwards ignoring the first row
for i in range(len(df_array)-1, 0, -1):
    if pd.isnull(df_array[i][3]):
        df_array[i-1][2] += " "+ df_array[i][2]
        df_array = np.delete(df_array, i, 0)

print("Converting to list of tuples...")
train_data=[]
for entity in df_array:
    location=entity[1].find(entity[2])
    length=len(entity[2])
    train_data.append(
        (entity[1], [(location, location+length,entity[3])])
    )

print("Prepping for DocBin...")

pre_doc_bin=[]
last_string = this_string = train_data[i][0]
current_entity_list = []

for i in range(len(train_data)):
    this_string = train_data[i][0]
    this_start_index = train_data[i][1][0][0]
    this_end_index = train_data[i][1][0][1]
    this_tag = train_data[i][1][0][2]

    if this_string == last_string:
        current_entity_list.append((this_start_index, this_end_index, this_tag))
    else:
        pre_doc_bin.append((last_string, current_entity_list))
        current_entity_list = []
        current_entity_list.append((this_start_index, this_end_index, this_tag))
        last_string = this_string
    #handle the last element as a special case
    if i == len(train_data)-1:
        pre_doc_bin.append((last_string, current_entity_list))

print("Prepped data sample: ")
print(pre_doc_bin[0:3])
print("Length of pre_doc_bin: ", len(pre_doc_bin))

import spacy
from spacy.tokens import DocBin
nlp = spacy.blank("en")
db = DocBin()
for text, annotations in pre_doc_bin:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations:
        span = doc.char_span(start, end, label=label)
        ents.append(span)
    #print(doc.ents)
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