#pip install gensim
#pip install nltk
#pip install tqdm

#!/usr/bin/env python
# -*- coding: UTF-8

# Word Embedding Models: Preprocessing and Word2Vec Model Training
# Project title: Charter school identities 
# Creators: Yoon Sung Hong and Jaren Haber
# Institution: Department of Sociology, University of California, Berkeley
# Date created: June 9, 2019
# Date last edited: February 3, 2022

# Import general packages
import imp, importlib # For working with modules
import nltk # for natural language processing tools
import pandas as pd # for working with dataframes
#from pandas.core.groupby.groupby import PanelGroupBy # For debugging
import numpy as np # for working with numbers
import pickle # For working with .pkl files
from tqdm import tqdm # Shows progress over iterations, including in pandas via "progress_apply"
import sys # For terminal tricks
import _pickle as cPickle # Optimized version of pickle
import gc # For managing garbage collector
import timeit # For counting time taken for a process
import datetime # For workin g with dates & times

# Import packages for cleaning, tokenizing, and stemming text
import re # For parsing text
from unicodedata import normalize # for cleaning text by converting unicode character encodings into readable format
from nltk import word_tokenize, sent_tokenize # widely used text tokenizer
from nltk.stem.porter import PorterStemmer # an approximate method of stemming words (it just cuts off the ends)
from nltk.stem.porter import PorterStemmer # approximate but effective (and common) method of normalizing words: stems words by implementing a hierarchy of linguistic rules that transform or cut off word endings
stem = PorterStemmer().stem # Makes stemming more accessible
from nltk.corpus import stopwords # for eliminating stop words
import gensim # For word embedding models
from gensim.models.phrases import Phrases # Makes word2vec more robust: Looks not just at  To look for multi-word phrases within word2vec

import string # for one method of eliminating punctuation
from nltk.corpus import stopwords # for eliminating stop words
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer; ps = PorterStemmer() # approximate but effective (and common) method of stemming words

#setting up multiprocessing
import multiprocessing
from sklearn import utils
cores = multiprocessing.cpu_count()

# Import packages for multiprocessing
import os # For navigation
numcpus = len(os.sched_getaffinity(0)) # Detect and assign number of available CPUs
from multiprocessing import Pool # key function for multiprocessing, to increase processing speed
pool = Pool(processes=numcpus) # Pre-load number of CPUs into pool function
import Cython # For parallelizing word2vec
mpdo = False # Set to 'True' if using multiprocessing--faster for creating words by sentence file, but more complicated
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')


from os import listdir
from os.path import isfile, join

import sys; sys.path.insert(0, "../../classification/preprocess/")
from clean_text import stopwords_make, punctstr_make, unicode_make, clean_sentence_apache

cwd = os.getcwd()
cwd = cwd.replace('embeddings/word2vec', 'jstor_data/ocr')
files = ['../../jstor_data/ocr/' + f for f in listdir(cwd) if isfile(join(cwd, f))]

#initializing two lists for strings from files and the filenames
text_ls = []
filename_ls = []
for file in files: #using sample only for cmputational speed purposes, change files_sample --> files for script
    with open(file, 'r') as myfile:
        data = myfile.read()
    data = data.replace('<plain_text><page sequence="1">', '')
    data = re.sub(r'</page>(\<.*?\>)', ' \n ', data)
    text_ls.append(data)
    filename_ls.append(file.replace('../ocr/', ''))
    
print("Text Files Loading Complete!")

d = {'filename': filename_ls, 'text': text_ls}
df = pd.DataFrame(d)



# Prep dictionaries of English words
from nltk.corpus import words # Dictionary of 236K English words from NLTK
english_nltk = set(words.words()) # Make callable
english_long = set() # Dictionary of 467K English words from https://github.com/dwyl/english-words
fname =  "english_words.txt" # Set file path to long english dictionary
with open(fname, "r") as f:
    for word in f:
        english_long.add(word.strip())
        
def stopwords_make(vocab_path_old = "", extend_stopwords = False):
    """Create stopwords list. 
    If extend_stopwords is True, create larger stopword list by joining sklearn list to NLTK list."""
                                                     
    stop_word_list = list(set(stopwords.words("english"))) # list of english stopwords

    # Add dates to stopwords
    for i in range(1,13):
        stop_word_list.append(datetime.date(2008, i, 1).strftime('%B'))
    for i in range(1,13):
        stop_word_list.append((datetime.date(2008, i, 1).strftime('%B')).lower())
    for i in range(1, 2100):
        stop_word_list.append(str(i))

    # Add other common stopwords
    stop_word_list.append('00') 
    stop_word_list.extend(['mr', 'mrs', 'sa', 'fax', 'email', 'phone', 'am', 'pm', 'org', 'com', 
                           'Menu', 'Contact Us', 'Facebook', 'Calendar', 'Lunch', 'Breakfast', 
                           'facebook', 'FAQs', 'FAQ', 'faq', 'faqs']) # web stopwords
    stop_word_list.extend(['el', 'en', 'la', 'los', 'para', 'las', 'san']) # Spanish stopwords
    stop_word_list.extend(['angeles', 'diego', 'harlem', 'bronx', 'austin', 'antonio']) # cities with many charter schools

    # Add state names & abbreviations (both uppercase and lowercase) to stopwords
    states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 
              'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 
              'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 
              'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 
              'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WI', 'WV', 'WY', 
              'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 
              'Colorado', 'Connecticut', 'District of Columbia', 'Delaware', 'Florida', 
              'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 
              'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 
              'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 
              'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 
              'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 
              'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 
              'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 
              'Vermont', 'Virginia', 'Washington', 'Wisconsin', 'West Virginia', 'Wyoming' 
              'carolina', 'columbia', 'dakota', 'hampshire', 'mexico', 'rhode', 'york']
    for state in states:
        stop_word_list.append(state)
    for state in [state.lower() for state in states]:
        stop_word_list.append(state)
        
    # Add even more stop words:
    if extend_stopwords == True:
        stop_word_list = text.ENGLISH_STOP_WORDS.union(stop_word_list)
        
    # If path to old vocab not specified, skip last step and return stop word list thus far
    if vocab_path_old == "":
        return stop_word_list

    # Add to stopwords useless and hard-to-formalize words/chars from first chunk of previous model vocab (e.g., a3d0, \fs19)
    # First create whitelist of useful terms probably in that list, explicitly exclude from junk words list both these and words with underscores (common phrases)
    whitelist = ["Pre-K", "pre-k", "pre-K", "preK", "prek", 
                 "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th", 
                 "1st-grade", "2nd-grade", "3rd-grade", "4th-grade", "5th-grade", "6th-grade", 
                 "7th-grade", "8th-grade", "9th-grade", "10th-grade", "11th-grade", "12th-grade", 
                 "1st-grader", "2nd-grader", "3rd-grader", "4th-grader", "5th-grader", "6th-grader", 
                 "7th-grader", "8th-grader", "9th-grader", "10th-grader", "11th-grader", "12th-grader", 
                 "1stgrade", "2ndgrade", "3rdgrade", "4thgrade", "5thgrade", "6thgrade", 
                 "7thgrade", "8thgrade", "9thgrade", "10thgrade", "11thgrade", "12thgrade", 
                 "1stgrader", "2ndgrader", "3rdgrader", "4thgrader", "5thgrader", "6thgrader", 
                 "7thgrader", "8thgrader", "9thgrader", "10thgrader", "11thgrader", "12thgrader"]
    with open(vocab_path_old) as f: # Load vocab from previous model
        junk_words = f.read().splitlines() 
    junk_words = [word for word in junk_words[:8511] if ((not "_" in word) 
                                                         and (not any(term in word for term in whitelist)))]
    stop_word_list.extend(junk_words)
                                                     
    return stop_word_list
                                                     
    
def punctstr_make():
    """Creates punctuations list"""
                    
    punctuations = list(string.punctuation) # assign list of common punctuation symbols
    #addpuncts = ['*','•','©','–','`','’','“','”','»','.','×','|','_','§','…','⎫'] # a few more punctuations also common in web text
    #punctuations += addpuncts # Expand punctuations list
    #punctuations = list(set(punctuations)) # Remove duplicates
    punctuations.remove('-') # Don't remove hyphens - dashes at beginning and end of words are handled separately)
    punctuations.remove("'") # Don't remove possessive apostrophes - those at beginning and end of words are handled separately
    punctstr = "".join([char for char in punctuations]) # Turn into string for regex later

    return punctstr
                                                     
                                                     
def unicode_make():
    """Create list of unicode chars"""
                    
    unicode_list  = []
    for i in range(1000,3000):
        unicode_list.append(chr(i))
    unicode_list.append("_cid:10") # Common in webtext junk
                                                     
    return unicode_list


def get_common_words(tokenized_corpus, max_percentage):
    """Discover most common words in corpus up to max_percentage.
    
    Args:
        Corpus tokenized by words,
        Highest allowable frequency of documents in which a token may appear (e.g., 1-5%)
        
    Returns:
        List of most frequent words in corpus"""
    
    # Code goes here
    # Probably using nltk.CountVectorizer

def write_list(file_path, textlist):
    """Writes textlist to file_path. Useful for recording output of parse_school()."""
    
    with open(file_path, 'w') as file_handler:
        
        for elem in textlist:
            file_handler.write("{}\n".format(elem))
    
    return    
    
# Create useful lists using above functions:
stop_words_list = stopwords_make()
punctstr = punctstr_make()
unicode_list = unicode_make()

def clean_sentence(sentence, remove_stopwords = True, keep_english = False, fast = False, exclude_words = [], stemming=False):
    """Removes numbers, emails, URLs, unicode characters, hex characters, and punctuation from a sentence 
    separated by whitespaces. Returns a tokenized, cleaned list of words from the sentence.
    
    Args: 
        sentence, i.e. string that possibly includes spaces and punctuation
        remove_stopwords: whether to remove stopwords, default True
        keep_english: whether to remove words not in english dictionary, default False; if 'restrictive', keep word only if in NLTK's dictionary of 237K english words; if 'permissive', keep word only if in longer list of 436K english words
        fast: whether to skip advanced sentence cleaning, removing emails, URLs, and unicode and hex chars, default False
        exclude_words: list of words to exclude, may be most common words or named entities, default empty list
        stemming: whether to apply PorterStemmer to each word, default False
    Returns: 
        Cleaned & tokenized sentence, i.e. a list of cleaned, lower-case, one-word strings"""
    
    global stop_words_list, punctstr, unicode_list, english_nltk, english_long
    
    # Replace unicode spaces, tabs, and underscores with spaces, and remove whitespaces from start/end of sentence:
    sentence = sentence.replace(u"\xa0", u" ").replace(u"\\t", u" ").replace(u"_", u" ").strip(" ")
    
    if not fast:
        # Remove hex characters (e.g., \xa0\, \x80):
        sentence = re.sub(r'[^\x00-\x7f]', r'', sentence) #replace anything that starts with a hex character 

        # Replace \\x, \\u, \\b, or anything that ends with \u2605
        sentence = re.sub(r"\\x.*|\\u.*|\\b.*|\u2605$", "", sentence)

        # Remove all elements that appear in unicode_list (looks like r'u1000|u10001|'):
        sentence = re.sub(r'|'.join(map(re.escape, unicode_list)), '', sentence)
    
    sentence = re.sub("\d+", "", sentence) # Remove numbers
    
    sent_list = [] # Initialize empty list to hold tokenized sentence (words added one at a time)
    
    for word in sentence.split(): # Split by spaces and iterate over words
        
        word = word.strip() # Remove leading and trailing spaces
        
        # Filter out emails and URLs:
        if not fast and ("@" in word or word.startswith(('http', 'https', 'www', '//', '\\', 'x_', 'x/', 'srcimage')) or word.endswith(('.com', '.net', '.gov', '.org', '.jpg', '.pdf', 'png', 'jpeg', 'php'))):
            continue
            
        # Remove punctuation (only after URLs removed):
        word = re.sub(r"["+punctstr+"]+", r'', word).strip("'").strip("-") # Remove punctuations, and remove dashes and apostrophes only from start/end of words
        
        if remove_stopwords and word in stop_words_list: # Filter out stop words
            continue
                
        # TO DO: Pass in most_common_words to function; write function to find the top 1-5% most frequent words, which we will exclude
        # Remove most common words:
        if word in exclude_words:
            continue
            
        if keep_english == 'restrictive':
            if word not in english_nltk: #Filter out non-English words using shorter list
                continue
            
        if keep_english == 'permissive': 
            if word not in english_long: #Filter out non-English words using longer list
                continue
        
        # Stem word (if applicable):
        if stemming:
            word = ps.stem(word)
        
        sent_list.append(word.lower()) # Add lower-cased word to list (after passing checks)

    return sent_list # Return clean, tokenized sentence

# ## Create lists of stopwords, punctuation, and unicode characters
stop_words_list = stopwords_make() # Define old vocab file path if you want to remove first, dirty elements
unicode_list = unicode_make()
punctstr = punctstr_make()

print("Stopwords, Unicodes, Punctuations lists creation complete!")


#word2vec computation
whole_text = []
s_count = 0 #initializing count for number of schools' texts appended
for school in df['text']:
    s_count += 1
    if s_count % 10000 == 0:
        print("Processed: ", s_count, " Schools' texts.")
    for chunk in school.split("\n"):
        for sent in sent_tokenize(chunk):
            sent = clean_sentence_apache(sent, unhyphenate=True, remove_propernouns=False)
            sent = [word for word in sent if word != '']
            if len(sent) > 0:
                whole_text.append(sent)

print("Text appending/processing complete!")

#defining directory locations to save word embedding model/vocab
cwd = os.getcwd()
cwd = cwd.replace('embeddings/word2vec', 'models_storage/word_embeddings_data')
model_path = cwd + "/word2vec_phrased_filtered_300d_2022_feb.bin"
vocab_path = cwd + "/wem_vocab_phrased_filtered_300d_2022_feb.txt" #named _phrased. Remove if you don't want to use phrases

# Train the model with above parameters:
print("Training word2vec model...") #change the words_by_sentence below to whole text if you don't want to use phrases
model = gensim.models.Word2Vec(words_by_sentence, size=300, window=10, min_count=5, sg=1, alpha=0.05,\
                               iter=50, batch_words=10000, workers=cores, seed=0, negative=5, ns_exponent=0.75)
print("word2vec model TRAINED successfully!")


# Save model:
from gensim.test.utils import get_tmpfile

fname = "word2vec_phrased_filtered_300d_2022_feb.bin"
model.save(fname)
print("Model Saved!")

#For reference
#model.wv.save_word2vec_format("wem_model_phrased_filtered_300d.bin", binary=True)
#model.save("wem_vocab_phrased_filtered_300d.txt")
#model.wv.save_word2vec_format(model_path, binary=True)
#model.save(vocab_path)
               
# Load word2vec model and save vocab list
#model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
#write_list(vocab_path, sorted(list(model.vocab)))
#print("word2vec model VOCAB saved to " + str(vocab_path))


            
            

