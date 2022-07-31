import json
import sys
import logging
import os
import pandas as pd
import csv



# def tsv_to_better_tsv(input_path, output_path):
#     df = pd.read
    
#     # try:
#     #     f=open(input_path,'r', encoding = "utf8") # input file
#     #     fp=open(output_path, 'w') # output file
#     #     for line in f:
#     #         if line[0:len(line)-1]!='.\tO':
#     #             word,entity=line.split()
#     #             fp.write(word+"\t"+entity[:len(entity)-1]+"\n")
#     #         else:
#     #             fp.write(line)
#     # except Exception as e:
#     #     logging.exception("Unable to process file" + "\n" + "error = " + str(e))
#     #     return None

def tsv_to_json_format(input_path,output_path,unknown_label):
    try:
        f=open(input_path,'r', encoding = "utf8") # input file
        fp=open(output_path, 'w') # output file
        data_dict={}
        annotations =[]
        label_dict={}
        s=''
        start=0
        for line in f:
            if line[0:len(line)-1]!='.\tO':
                word,entity=line.split()
                s+=word+" "
                entity=entity[:len(entity)-1]
                if entity!=unknown_label:
                    if len(entity) != 1:
                        d={}
                        d['text']=word
                        d['start']=start
                        d['end']=start+len(word)-1  
                        try:
                            label_dict[entity].append(d)
                        except:
                            label_dict[entity]=[]
                            label_dict[entity].append(d) 
                start+=len(word)+1
            else:
                data_dict['content']=s
                s=''
                label_list=[]
                for ents in list(label_dict.keys()):
                    for i in range(len(label_dict[ents])):
                        if(label_dict[ents][i]['text']!=''):
                            l=[ents,label_dict[ents][i]]
                            for j in range(i+1,len(label_dict[ents])): 
                                if(label_dict[ents][i]['text']==label_dict[ents][j]['text']):  
                                    di={}
                                    di['start']=label_dict[ents][j]['start']
                                    di['end']=label_dict[ents][j]['end']
                                    di['text']=label_dict[ents][i]['text']
                                    l.append(di)
                                    label_dict[ents][j]['text']=''
                            label_list.append(l)                          
                            
                for entities in label_list:
                    label={}
                    label['label']=[entities[0]]
                    label['points']=entities[1:]
                    annotations.append(label)
                data_dict['annotation']=annotations
                annotations=[]
                json.dump(data_dict, fp)
                fp.write('\n')
                data_dict={}
                start=0
                label_dict={}
    except Exception as e:
        logging.exception("Unable to process file" + "\n" + "error = " + str(e))
        return None

#tsv_to_json_format("Train_Tagged_Titles.tsv",'ner_corpus_260.json','abc')

tsv_to_better_tsv("Train_Tagged_Titles.tsv","Train_Tagged_Titles_better.tsv")
