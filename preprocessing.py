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
        
        (entity[1], {"entities": [(location, location+length,entity[3])]})
    )

from __future__ import unicode_literals, print_function
import pickle

import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding

LABEL = ['Accents', 'Brand', 'Character', 'Character Family', 'Closure', 'Color', 'Country/Region of Manufacture', 'Department', 'Fabric Type', 'Features', 'Handle Drop', 'Handle Style']

def main(model=None, new_model_name='new_model', output_dir=None, n_iter=10):
    """Setting up the pipeline and entity recognizer, and training the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spacy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    else:
        ner = nlp.get_pipe('ner')

    for i in LABEL:
        ner.add_label(i)   # Add new entity labels to entity recognizer

    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.entity.create_optimizer()

    # Get names of other pipes to disable them during training to train only NER
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                           losses=losses)
            print('Losses', losses)

    # Test the trained model
    test_text = 'Gianni Infantino is the president of FIFA.'
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # Save model 
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # Test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)






#print df.head 