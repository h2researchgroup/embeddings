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

# get base file paths
root = getcwd() # current dir
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
all_prepped_fp = prepped_fp + 'filtered_enchant_orgdict_preprocessed_texts_59098_121722.pkl'
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
#                  Load data                  #
###############################################

# Load preprocessed text data
words_by_sentence_ALL = quickpickle.load(all_prepped_fp)
words_by_sentence_d1 = quickpickle.load(d1_prepped_fp)
words_by_sentence_d2 = quickpickle.load(d2_prepped_fp)
words_by_sentence_d3 = quickpickle.load(d3_prepped_fp)
words_by_sentence_d4 = quickpickle.load(d4_prepped_fp)


###############################################
#            Train Word2Vec models            #
###############################################

# Note: numdims (size of vector space) and windows (word window size) are defined at top of 'Initialize' section above
print("Training word2vec models...") 

model_d1 = Word2Vec(words_by_sentence_d1, vector_size=numdims, window=windows, min_count=5, sg=1, alpha=0.05,
                      epochs=50, batch_words=10000, workers=cores, seed=0, negative=5, ns_exponent=0.75)
d1_model.save(model_d1_fp)
print("Decade 1 model trained & saved as sanity check! Training decade 2 model...")
model_d2 = Word2Vec(words_by_sentence_d2, vector_size=numdims, window=windows, min_count=5, sg=1, alpha=0.05,
                      epochs=50, batch_words=10000, workers=cores, seed=0, negative=5, ns_exponent=0.75)
print("Decade 2 model trained! Training decade 3 model...")
model_d3 = Word2Vec(words_by_sentence_d3, vector_size=numdims, window=windows, min_count=5, sg=1, alpha=0.05,
                      epochs=50, batch_words=10000, workers=cores, seed=0, negative=5, ns_exponent=0.75)
print("Decade 3 model trained! Training decade 4 model...")
model_d4 = Word2Vec(words_by_sentence_d4, vector_size=numdims, window=windows, min_count=5, sg=1, alpha=0.05,
                      epochs=50, batch_words=10000, workers=cores, seed=0, negative=5, ns_exponent=0.75)
print("Decade 4 model trained! Training full model...")
full_model = Word2Vec(words_by_sentence_ALL, vector_size=numdims, window=windows, min_count=5, sg=1, alpha=0.05,
                      epochs=50, batch_words=10000, workers=cores, seed=0, negative=5, ns_exponent=0.75)

print("All word2vec models trained successfully!")


###############################################
#                 Save models                 #
###############################################

#d1_model.save(model_d1_fp)
d2_model.save(model_d2_fp)
d3_model.save(model_d3_fp)
d4_model.save(model_d4_fp)
full_model.save(model_full_fp)

print("Models Saved!")

sys.exit() # Close script to be safe
