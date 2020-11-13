from gensim.test.utils import get_tmpfile
import os
import numpy as np
import pandas as pd
import gensim


cwd = os.getcwd()

import tqdm
from os import listdir
from os.path import isfile, join
import re
import sys; sys.path.insert(0, "../../../data_management/tools/")
from clean_text import stopwords_make, punctstr_make, unicode_make, get_common_words, clean_sentence_apache

print("Imports completed!")

ocr_wd = cwd.replace('Computational-Analysis-For-Social-Science/WordEmbedding/other_scripts', 'jstor_data/ocr/')
colnames = ['file_name']
articles = pd.read_csv("../../../models_storage/word_embeddings_data/filtered_index.csv", names=colnames, header=None)
files_to_be_opened = ["../../../jstor_data/ocr/" + file + '.txt' for file in articles.file_name]
all_files = ['../../../jstor_data/ocr/' + f for f in listdir(ocr_wd) if isfile(join(ocr_wd, f))]

files = [file for file in all_files if file in files_to_be_opened]

#initializing two lists for strings from files and the filenames
text_ls = []
filename_ls = []
index = 1
for file in files:
    with open(file, 'r') as myfile:
        data = myfile.read()
    data = data.replace('<plain_text><page sequence="1">', '')
    data = re.sub(r'</page>(\<.*?\>)', ' \n ', data)
    data = clean_sentence_apache(data, unhyphenate=True, remove_propernouns=False, remove_acronyms=False, return_string=True)
    text_ls.append(data)
    filename_ls.append(file.replace('../ocr/', ''))
    if index % 1000 == 0:
        print("Cleaned ", index, " documents.") 
    index += 1

print("Text Cleaning completed!")

d = {'filename': filename_ls, 'text': text_ls}
df = pd.DataFrame(d)

print("Shortening texts...")

df["edited_filename"] = df['filename'].apply(lambda x: x[40:-4])
df.text = df.text.apply(lambda x: x[:10000] if len(x) > 10000 else x) #cutting down to 10000 words max

df.to_csv("../../../models_storage/word_embeddings_data/ocr_text_with_tags_10000_2020_sept5.csv")
print("Saved the data-frame with cleaned, truncated texts!")