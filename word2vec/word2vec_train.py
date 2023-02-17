#!/usr/bin/env python
# -*- coding: UTF-8

'''
@title: Train word embedding models with word2vec
@author: Jaren Haber, PhD, Dartmouth College
@coauthors: Prof. Heather Haveman, UC Berkeley; Yoon Sung Hong, Wayfair
@contact: jhaber@berkeley.edu
@project: Computational Literature Review of Organizational Scholarship
@repo: https://github.com/h2researchgroup/classification/
@date: December 15, 2020
@description: Trains word embedding models using word2vec preprocessed JSTOR article text data from 1971-2014. Trains on full and decade-specific text data.
'''

###############################################
#                  Initialize                 #
###############################################

# set params for w2v models
numdims = 300 # number of dimensions in vector space model
windows = 10 # word window size

# Import general packages
import imp, importlib # For working with modules
import pandas as pd # for working with dataframes
import numpy as np # for working with numbers
import pickle # For working with .pkl files
from tqdm import tqdm # Shows progress over iterations, including in pandas via "progress_apply"
import sys # For terminal tricks
import _pickle as cPickle # Optimized version of pickle
import gc # For managing garbage collector
import timeit # For counting time taken for a process
from os import listdir, getcwd, sched_getaffinity
from os.path import isfile, join
from datetime import date # For working with dates & times
import re # For parsing text
from gensim.models import Word2Vec, KeyedVectors # for word embedding models
from multiprocessing import cpu_count; cores = cpu_count() # count cores

# get base file paths
cwd = getcwd() # current dir
root = str.replace(cwd, 'embeddings/word2vec', '')
thisday = date.today().strftime("%m%d%y")

# import custom functions
import sys; sys.path.insert(0, "../../classification/preprocess/")
from file_utils import quickpickle_dump, quickpickle_load, write_textlist # for quick saving & loading to pickle format


###############################################
#              Define file paths              #
###############################################

# paths to raw text files
jstor_fp = root + 'jstor_data/ocr'
files = [jstor_fp + f for f in listdir(jstor_fp) if isfile(join(jstor_fp, f))]

# path to preprocessed text data
prepped_fp = root + 'models_storage/preprocessed_texts/'
all_prepped_fp = prepped_fp + 'filtered_enchant_orgdict_preprocessed_texts_ALL_59098_121722.pkl'
d1_prepped_fp = prepped_fp + 'filtered_enchant_orgdict_preprocessed_texts_1971-1981_8014_121722.pkl'
d2_prepped_fp = prepped_fp + 'filtered_enchant_orgdict_preprocessed_texts_1982-1992_13245_121722.pkl'
d3_prepped_fp = prepped_fp + 'filtered_enchant_orgdict_preprocessed_texts_1993-2003_17566_121722.pkl'
d4_prepped_fp = prepped_fp + 'filtered_enchant_orgdict_preprocessed_texts_2004-2014_19825_121722.pkl'

# filepaths to save word embedding model/vocab
w2v_fp = root + 'models_storage/w2v_models/'
model_full_fp = w2v_fp + f"word2vec_ALLYEARS_phrased_filtered_enchant_orgdict_{numdims}d_{windows}w_{thisday}.bin"
model_d1_fp = w2v_fp + f"word2vec_1971-1981_phrased_filtered_enchant_orgdict_{numdims}d_{windows}w_{thisday}.bin"
model_d2_fp = w2v_fp + f"word2vec_1982-1992_phrased_filtered_enchant_orgdict_{numdims}d_{windows}w_{thisday}.bin"
model_d3_fp = w2v_fp + f"word2vec_1993-2003_phrased_filtered_enchant_orgdict_{numdims}d_{windows}w_{thisday}.bin"
model_d4_fp = w2v_fp + f"word2vec_2004-2014_phrased_filtered_enchant_orgdict_{numdims}d_{windows}w_{thisday}.bin"


###############################################
#              Define functions               #
###############################################

def prep_text(texts_fp:str, 
              text_col:str):
    '''Creates list of texts for use in training Word2Vec model. Each text is a list of terms in order. 
    Uses 'text' column in DataFrame located at texts_fp to create list.'''
    
    df = quickpickle_load(texts_fp)
    
    texts = []
    for row in df[text_col]:
        words_by_sentence = []
        for section in row: # each section is a paragraph or so
            words_by_sentence.extend(section) # add section to list; represent doc as list of terms
        texts.append(words_by_sentence) # add doc to full list

    return texts


def train_save_w2v(texts_fp: str, 
                   numdims: int, 
                   windows: int, 
                   model_save_fp:str):
    """
    Loads preprocessed text data, trains word2vec model with specified number of dimensions and windows, and 
    saves model to specified filepath. 
    
    Params:
        texts_fp (str): path to file with preprocessed texts
        numdims (int): number of dimensions for word2vec model
        windows (int): number of windows for word2vec model
        model_save_fp (str): path to file to save word2vec model
        
    Returns: 
        None (saves model to file)
    """
    
    print(f"Loading preprocessed text data from '{texts_fp.split('/')[-1]}'...")
    words_by_doc = prep_text(texts_fp, text_col='text')
    
    print("Training Word2Vec model...") 
    w2v_model = Word2Vec(words_by_doc, vector_size=numdims, window=windows, min_count=5, sg=1, alpha=0.05,
                         epochs=50, batch_words=10000, workers=cores, seed=0, negative=5, ns_exponent=0.75)
    
    w2v_model.save(model_save_fp)
    print(f"Word2Vec model trained and saved to {model_save_fp.split('/')[-1]}!")
    print()
        
    return


###############################################
#        Train & save Word2Vec models         #
###############################################

train_save_w2v(d1_prepped_fp, numdims=numdims, windows=windows, model_save_fp=model_d1_fp)
train_save_w2v(d2_prepped_fp, numdims=numdims, windows=windows, model_save_fp=model_d2_fp)
train_save_w2v(d3_prepped_fp, numdims=numdims, windows=windows, model_save_fp=model_d3_fp)
train_save_w2v(d4_prepped_fp, numdims=numdims, windows=windows, model_save_fp=model_d4_fp)
train_save_w2v(all_prepped_fp, numdims=numdims, windows=windows, model_save_fp=model_full_fp)


sys.exit() # Close script to be safe
