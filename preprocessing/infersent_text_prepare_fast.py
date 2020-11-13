from gensim.test.utils import get_tmpfile
import os
import numpy as np
import pandas as pd
import gensim

import tqdm
from os import listdir
from os.path import isfile, join
import re
import sys; sys.path.insert(0, "../../../data_management/tools/")
from clean_text import stopwords_make, punctstr_make, unicode_make, get_common_words, clean_sentence_apache
from quickpickle import quickpickle_load

print("Imports completed!")

cwd = os.getcwd()
ocr_wd = cwd.replace('Computational-Analysis-For-Social-Science/WordEmbedding/other_scripts', 'jstor_data/ocr')
#files = ['../../../jstor_data/ocr/' + f for f in listdir(cwd) if isfile(join(cwd, f))]
colnames = ['file_name']
articles = pd.read_csv("../../../models_storage/word_embeddings_data/filtered_index.csv", names=colnames, header=None)
files_to_be_opened = ["../../../jstor_data/ocr/" + file + '.txt' for file in articles.file_name]
all_files = ['../../../jstor_data/ocr/' + f for f in listdir(ocr_wd) if isfile(join(ocr_wd, f))]

files = [file for file in all_files if file in files_to_be_opened]

print("Loading pickle file of the nested, cleaned sentences...")
whole_text = quickpickle_load("../../../models_storage/word_embeddings_data/cleaned_text_flat_nov21.pkl")
print("Pickle file loaded as nested list!")

#initializing two lists for strings from files and the filenames
filename_ls = []
for file in files: #using sample only for cmputational speed purposes, change files_sample --> files for script
    try:
        with open(file, 'r') as myfile:
            data = myfile.read()
        filename_ls.append(file[40:-4])
    except:
        print(file[40:-4], ", doesn't exist in ocr folder. Passing...")
        
text = [' '.join(ls) for ls in whole_text]

d = {'filename': filename_ls, 'text': text}
df = pd.DataFrame(d)

print("Shortening texts...")

df["edited_filename"] = df['filename'].apply(lambda x: x[40:-4])
df.text = df.text.apply(lambda x: x[:10000] if len(x) > 10000 else x) #cutting down to 10000 words max
        
df.to_csv("../data/ocr_text_with_tags_10000_dec11.csv")
print("Saved the data-frame with cleaned, truncated texts!")