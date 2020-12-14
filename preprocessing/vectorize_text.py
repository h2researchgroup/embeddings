#!/usr/bin/env python
# coding: utf-8

'''
@title: Preprocess Articles for JSTOR Classifier Training
@author: Jaren Haber, PhD, Georgetown University
@coauthors: Prof. Heather Haveman, UC Berkeley; Yoon Sung Hong, Wayfair
@contact: Jaren.Haber@georgetown.edu
@project: Computational Literature Review of Organizational Scholarship
@repo: https://github.com/h2researchgroup/classification/
@date: December 2020
@description: Preprocesses article data for classifier training purposes by vectorizing preprocessed text into DTMs with and without TF-IDF weighting. Saves DTMs for both truncated version of each article (first 500 words only) and complete versions.
'''


###############################################
# Initialize
###############################################

# import packages
import imp, importlib # For working with modules
import pandas as pd # for working with dataframes
import numpy as np # for working with numbers
import pickle # For working with .pkl files
import re # for regex magic
from tqdm import tqdm # Shows progress over iterations, including in pandas via "progress_apply"
import sys # For terminal tricks
import csv
import _pickle as cPickle # Optimized version of pickle
import gc # For managing garbage collector
import timeit # For counting time taken for a process
from datetime import date # For working with dates & times
from nltk import sent_tokenize
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import tables
import random
import os; from os import listdir; from os.path import isfile, join

# Define base file path
cwd = os.getcwd()
root = str.replace(cwd, 'embeddings_jstor/preprocessing', '') # retains last slash

# Custom scripts for working with texts in Python
import sys; sys.path.insert(0, root + "classification/preprocess/") # For loading functions from files in other directory
from clean_text import stopwords_make, punctstr_make, unicode_make, apache_tokenize, clean_sentence_apache # for preprocessing text
from quickpickle import quickpickle_dump, quickpickle_load # for quick saving & loading to pickle format
from text_to_file import write_textlist, read_text # custom scripts for reading and writing text lists to .txt files


###############################################
# Define file paths
###############################################

thisday = date.today().strftime("%m%d%y")

# directory for prepared data: save files here
data_fp = root + 'models_storage/word_embeddings_data/'
model_fp = root + 'models_storage/models/'

# Current article lists
article_list_fp = data_fp + 'filtered_length_index.csv' # Filtered index of research articles
article_paths_fp = data_fp + 'filtered_length_article_paths.csv' # List of article file paths

# Set number of words to keep from front of each article, akin to an abstract
# (This is in addition to processing and saving the complete texts)
keepfirst = 300


###############################################
# Load data
###############################################

# Read full list of articles for new sample selection
tqdm.pandas(desc='Correcting file paths...')
articles = (pd.read_csv(article_paths_fp, low_memory=False, header=None, names=['file_name']))
articles['file_name'] = articles['file_name'].progress_apply(lambda fp: re.sub('/home/jovyan/work/', root, fp)) # make sure base file name is correct

# Read text data from files
tqdm.pandas(desc='Loading ALL text files...')
articles['text'] = articles['file_name'].progress_apply(lambda fp: read_text(fp, shell = True))

# Create copy for truncated version of text (only first 500 words)
articles_truncated = articles.copy()

# Use articles data to define additional file paths
prepped_fp = data_fp + f'filtered_preprocessed_texts_{str(len(articles))}_{str(thisday)}.pkl' # ALL JSTOR preprocessed text
trunc_prepped_fp = data_fp + f'filtered_preprocessed_texts_{str(keepfirst)}_{str(len(articles))}_{str(thisday)}.pkl' # ALL JSTOR preprocessed text, some number of words (=keepfirst) words from front of each

vec_fp = model_fp + f'vectorizer_unweighted_{str(len(articles))}_{str(thisday)}.joblib' # unweighted vectorizer trained on ALL article text data 
tfidf_vec_fp = model_fp + f'vectorizer_tfidf_{str(len(articles))}_{str(thisday)}.joblib' # TF-IDF weighted vectorizer trained on ALL article text data 
vec_trunc_fp = model_fp + f'vectorizer_unweighted_trunc_{str(keepfirst)}_{str(len(articles))}_{str(thisday)}.joblib' # unweighted vectorizer trained on ALL article text data 
tfidf_vec_trunc_fp = model_fp + f'vectorizer_tfidf_trunc_{str(keepfirst)}_{str(len(articles))}_{str(thisday)}.joblib' # TF-IDF weighted vectorizer trained on ALL article text data 
vec_feat_fp = model_fp + f'vectorizer_features_{str(len(articles))}_{str(thisday)}.csv' # vocab of (unweighted) vectorizer (for verification purposes)
vec_feat_trunc_fp = model_fp + f'vectorizer_features_trunc_{str(keepfirst)}_{str(len(articles))}_{str(thisday)}.csv' # vocab of (unweighted) vectorizer, text truncated to keepfirst

dtm_fp = model_fp + f'dtm_unweighted_full_{str(len(articles))}_{str(thisday)}.pkl' # unweighted DTM, complete texts
tfidf_dtm_fp = model_fp + f'dtm_tfidf_full_{str(len(articles))}_{str(thisday)}.pkl' # TF-IDF weighted DTM, complete texts
dtm_trunc_fp = model_fp + f'dtm_unweighted_trunc_{str(keepfirst)}_{str(len(articles))}_{str(thisday)}.pkl' # unweighted DTM, truncated to keepfirst
dtm_tfidf_trunc_fp = model_fp + f'dtm_tfidf_trunc_{str(keepfirst)}_{str(len(articles))}_{str(thisday)}.pkl' # TF-IDF weighted DTM, truncated to keepfirst


###############################################
# Preprocess text files
###############################################

def preprocess_text(article, 
                    truncate = False, 
                    return_string = False):
    '''
    Cleans up articles by removing page marker junk, 
    unicode formatting, and extra whitespaces; 
    re-joining words split by (hyphenated at) end of line; 
    removing numbers (by default) and acronyms (not by default); 
    tokenizing sentences into words using the Apache Lucene Tokenizer (same as JSTOR); 
    lower-casing words; 
    removing stopwords (same as JSTOR), junk formatting words, junk sentence fragments, 
    and proper nouns (the last not by default).
    
    Args:
        article (str): raw OCR'd academic article text
        truncate (False or integer): whether to keep only first num words from each article (like an abstract)
        return_string (binary): whether to return str (instead of list of str)
        
    Returns:
        str or list of str: each element of list is a word
    '''
    
    if truncate:
        return_string_temp = False # need to return tokenized version to count words in article
    else:
        return_string_temp = return_string
    
    # Remove page marker junk
    article = article.replace('<plain_text><page sequence="1">', '')
    article = re.sub(r'</page>(\<.*?\>)', ' \n ', article)
    
    article = clean_sentence_apache(article, 
                                    unhyphenate=True, 
                                    remove_numbers=False, 
                                    remove_acronyms=False, 
                                    remove_stopwords=False, 
                                    remove_propernouns=False, 
                                    return_string=return_string_temp
                                    )
    
    if truncate:
        article = article[:truncate] # keep only num (=truncate) first words in each article
    
    if truncate and return_string: # join into string here if not earlier
        article = ' '.join(article)
        
    return article

tqdm.pandas(desc='Cleaning text files...')
articles['text'] = articles['text'].progress_apply(
    lambda text: preprocess_text(text, 
                                 return_string = True))
articles_truncated['text'] = articles_truncated['text'].progress_apply(
    lambda text: preprocess_text(text, 
                                 truncate = keepfirst, 
                                 return_string = True))

# Collect articles into one big list of strings, one per article, to train in vectorizers
tqdm.pandas(desc='Collecting sentences for vectorization...')
docs = []; articles['text'].progress_apply(lambda article: docs.append(article))
docs_trunc = []; articles_truncated['text'].progress_apply(lambda article: docs_trunc.append(article))


###############################################
# Apply vectorizers to create DTMs
###############################################

# Define stopwords used by JSTOR
jstor_stopwords = set(["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"])

print("Preparing and applying vectorizers...")

# Use TFIDF weighted DTM because results in better classifier accuracy than unweighted
vectorizer = CountVectorizer(max_features=100000, min_df=1, max_df=0.8, stop_words=jstor_stopwords) # unweighted DTM
tfidf_vectorizer = TfidfVectorizer(max_features=100000, min_df=1, max_df=0.8, stop_words=jstor_stopwords) # TFIDF-weighted DTM

# Create unweighted vectorizer and save its vocab (for verification purposes)
dtm = vectorizer.fit_transform(docs)
joblib.dump(vectorizer, open(vec_fp, "wb"))
with open(vec_feat_fp,'w') as f:
    writer = csv.writer(f)
    writer.writerows([vectorizer.get_feature_names()])
    
print('Number of features in vectorizer:', len(vectorizer.get_feature_names()))
    
# Create TF-IDF vectorizer (features are the same)
dtm_tfidf = tfidf_vectorizer.fit_transform(docs)
joblib.dump(tfidf_vectorizer, open(tfidf_vec_fp, "wb"))

# Repeat for truncated version of articles, start with new vectorizers to avoid duplication
vectorizer = CountVectorizer(max_features=100000, min_df=1, max_df=0.8, stop_words=jstor_stopwords) # unweighted DTM
tfidf_vectorizer = TfidfVectorizer(max_features=100000, min_df=1, max_df=0.8, stop_words=jstor_stopwords) # TFIDF-weighted DTM

dtm_trunc = vectorizer.fit_transform(docs_trunc)
joblib.dump(vectorizer, open(vec_trunc_fp, "wb"))
with open(vec_feat_trunc_fp,'w') as f:
    writer = csv.writer(f)
    writer.writerows([vectorizer.get_feature_names()])
    
print('Number of features in truncated vectorizer:', len(vectorizer.get_feature_names()))
    
# Create TF-IDF vectorizer (features are the same)
dtm_tfidf_trunc = tfidf_vectorizer.fit_transform(docs_trunc)
joblib.dump(tfidf_vectorizer, open(tfidf_vec_trunc_fp, "wb"))


###############################################
# Save DTMs and preprocessed text data
###############################################

print("Saving DTMs and preprocessed texts to file.")

# DTMs
quickpickle_dump(dtm, dtm_fp)
quickpickle_dump(dtm_tfidf, tfidf_dtm_fp)
quickpickle_dump(dtm_trunc, dtm_trunc_fp)
quickpickle_dump(dtm_tfidf_trunc, dtm_tfidf_trunc_fp)

# Preprocessed text data
quickpickle_dump(articles_truncated, trunc_prepped_fp)
quickpickle_dump(articles, prepped_fp)

print("Done saving preparatory objects for NLP.")

sys.exit() # Close script to be safe