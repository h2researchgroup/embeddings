from gensim.models import Doc2Vec
from gensim.test.utils import get_tmpfile
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import gensim
from numpy import dot, absolute
from numpy.linalg import norm
from gensim.models import Word2Vec
from tqdm import tqdm

cwd = os.getcwd()
cwd = cwd.replace("Computational-Analysis-For-Social-Science/WordEmbedding/other_scripts", 
                  "models_storage/word_embeddings_data/")
fname_dmm = get_tmpfile(cwd + "/dmm_model_phrased_filtered_oct15") #change if model updated
#fname_dmm = get_tmpfile("../../../models_storage/word_embeddings_data/dmm_model_phrased_filtered_sept2)
model_dmm = Doc2Vec.load(fname_dmm)

model_word2vec = gensim.models.KeyedVectors.load("../../../models_storage/word_embeddings_data/word2vec_phrased_filtered_300d_2020_oct17_1970.bin") #change if model updated
#model_word2vec = gensim.models.KeyedVectors.load("../../../models_storage/word_embeddings_data/word2vec_phrased_filtered_300d_aug14.bin)

from os import listdir
from os.path import isfile, join
import re

import sys; sys.path.insert(0, "../../../data_management/tools/")
from quickpickle import quickpickle_load

# df_infersent = pd.read_csv("../../../models_storage/word_embeddings_data/cosine_scores_infersent_10000.csv")


#or if you've already done the cosine calculations:
# text = pd.read_csv("ocr_text_with_tags_10000.csv")
# text = text[text.text.isna() == False]
text = pd.read_csv("../../../models_storage/word_embeddings_data/ocr_text_with_tags_10000_2020_sept5.csv")
df = pd.read_csv("../../../models_storage/word_embeddings_data/ocr_text_with_tags_10000_2020_sept5.csv")

ls_index_drop = []
for i in df.index:
    try:
        str.split(df[df.edited_filename == str(df.iloc[i,].edited_filename)].text.item())
    except:
           ls_index_drop.append(i)
            
df = df.drop(ls_index_drop)

#text_nested = quickpickle_load("cleaned_text_nested_aug14.pkl") # ## 
#this will change to the data frame cleaned by text_nest_list_save later. 
#or will we just keep it parallel to the infersent methods? worth clarifying with Jaren and Prof. Haveman.

# df = df_infersent.merge(text, how='left', on='edited_filename')

#Importing the GloVe model; takes ~5-10 minutes
#glove_vocab – list of the words that we now have embeddings for
#glove_embed – list of lists containing the embedding vectors
#embedding_dict – dictionary where the words are the keys and the embeddings are the values

filename = '../InferSent/encoder/glove.840B.300d.txt'
 
glove_vocab = []
glove_embed=[]
embedding_dict = {}
 
file = open(filename,'r',encoding='UTF-8')
 
for line in file.readlines():
    row = line.strip().split(' ')
    vocab_word = row[0]
    glove_vocab.append(vocab_word)
    embed_vector = [float(i) for i in row[1:]] # convert to list of float
    embedding_dict[vocab_word]=embed_vector
    glove_embed.append(embed_vector)

print("GloVe loaded!")
file.close()

#
# culture = pd.read_csv("../../../models_storage/word_embeddings_data/Culture_full.csv", sep='\n', header=None)
# culture.columns = ["vocab"]
# demographic = pd.read_csv("../../../models_storage/word_embeddings_data/Demographic_full.csv", sep='\n', header=None)
# demographic.columns = ["vocab"]
# relational = pd.read_csv("../../../models_storage/word_embeddings_data/Relational_full.csv", sep='\n', header=None)
# relational.columns = ["vocab"]


culture = pd.read_csv("../../Dictionary Mapping/Dictionaries/core/cultural_core_orgs.csv", sep='\n', header=None)
culture.columns = ["vocab"]
demographic = pd.read_csv("../../Dictionary Mapping/Dictionaries/core/demographic_core_orgs.csv", sep='\n', header=None)
demographic.columns = ["vocab"]
relational = pd.read_csv("../../Dictionary Mapping/Dictionaries/core/relational_core_orgs.csv", sep='\n', header=None)
relational.columns = ["vocab"]



# culture.vocab = culture.vocab.apply(lambda x: re.sub(',', '_', x))
# demographic.vocab = demographic.vocab.apply(lambda x: re.sub(',', '_', x))
# relational.vocab = relational.vocab.apply(lambda x: re.sub(',', '_', x))
#

#creating a nested dictionary structure to make key calls depending on the dict size and type
#structure goes down the following way - dictionary --> model type --> dictionary type --> dictionary size
dictionary = {}
word2vec_dict = {}
# doc2vec_dict = {}
# glove_dict = {}

word2vec_culture_dict = {}
word2vec_demographic_dict = {}
word2vec_relational_dict = {}
# doc2vec_culture_dict = {}
# doc2vec_demographic_dict = {}
# doc2vec_relational_dict = {}
# glove_culture_dict = {}
# glove_demographic_dict = {}
# glove_relational_dict = {}


###### Confirming Dictionaries in context of model types and vocab sizes
###### At the moment, only core size available. Later, add to this section other vocab sizes' cases too, using same dict calls.
culture_word2vec = []
culture_word2vec_emb = []

for word in culture.vocab:
    try:
        emb = model_word2vec[word]
        culture_word2vec.append(word)
        culture_word2vec_emb.append(emb)
    except:
        pass

print("Total # of culture terms: ", len(culture_word2vec))
culture_word2vec_emb = np.array(culture_word2vec_emb)
culture_word2vec_emb_mean = np.mean(culture_word2vec_emb, axis=0)
word2vec_culture_dict['core'] = culture_word2vec_emb_mean


demographic_word2vec = []
demographic_word2vec_emb = []

for word in demographic.vocab:
    try:
        emb = model_word2vec[word]
        demographic_word2vec.append(word)
        demographic_word2vec_emb.append(emb)
    except:
        pass

print("Total # of demographic terms: ", len(demographic_word2vec))
demographic_word2vec_emb = np.array(demographic_word2vec_emb)
demographic_word2vec_emb_mean = np.mean(demographic_word2vec_emb, axis=0)
word2vec_demographic_dict['core'] = demographic_word2vec_emb_mean


relational_word2vec = []
relational_word2vec_emb = []

for word in relational.vocab:
    try:
        emb = model_word2vec[word]
        relational_word2vec.append(word)
        relational_word2vec_emb.append(emb)
    except:
        pass

print("Total # of relational terms: ", len(relational_word2vec))
relational_word2vec_emb = np.array(relational_word2vec_emb)
relational_word2vec_emb_mean = np.mean(relational_word2vec_emb, axis=0)
word2vec_relational_dict['core'] = relational_word2vec_emb_mean


# culture_doc2vec = []
# culture_doc2vec_emb = []

# for word in culture.vocab:
#     try:
#         emb = model_dmm[word]
#         culture_doc2vec.append(word)
#         culture_doc2vec_emb.append(emb)
#     except:
#         pass

# print("Total # of culture terms: ", len(culture_doc2vec))
# culture_doc2vec_emb = np.array(culture_doc2vec_emb)
# culture_doc2vec_emb_mean = np.mean(culture_doc2vec_emb, axis=0)
# doc2vec_culture_dict['core'] = culture_doc2vec_emb_mean


# demographic_doc2vec = []
# demographic_doc2vec_emb = []

# for word in demographic.vocab:
#     try:
#         emb = model_dmm[word]
#         demographic_doc2vec.append(word)
#         demographic_doc2vec_emb.append(emb)
#     except:
#         pass

# print("Total # of demographic terms: ", len(demographic_doc2vec))
# demographic_doc2vec_emb = np.array(demographic_doc2vec_emb)
# demographic_doc2vec_emb_mean = np.mean(demographic_doc2vec_emb, axis=0)
# doc2vec_demographic_dict['core'] = demographic_doc2vec_emb_mean


# relational_doc2vec = []
# relational_doc2vec_emb = []
# for word in relational.vocab:
#     try:
#         emb = model_dmm[word]
#         relational_doc2vec.append(word)
#         relational_doc2vec_emb.append(emb)
#     except:
#         pass

# print("Total # of relational terms: ", len(relational_doc2vec))
# relational_doc2vec_emb = np.array(relational_doc2vec_emb)
# relational_doc2vec_emb_mean = np.mean(relational_doc2vec_emb, axis=0)
# doc2vec_relational_dict['core'] = relational_doc2vec_emb_mean


# culture_glove = []
# culture_glove_emb = []

# for word in culture.vocab:
#     try:
#         emb = embedding_dict[word]
#         culture_glove.append(word)
#         culture_glove_emb.append(emb)
#     except:
#         pass

# print("Total # of culture terms: ", len(culture_glove))
# culture_glove_emb = np.array(culture_glove_emb)
# culture_glove_emb_mean = np.mean(culture_glove_emb, axis=0)
# glove_culture_dict['core'] = culture_glove_emb_mean


# demographic_glove = []
# demographic_glove_emb = []

# for word in demographic.vocab:
#     try:
#         emb = embedding_dict[word]
#         demographic_glove.append(word)
#         demographic_glove_emb.append(emb)
#     except:
#         pass

# print("Total # of demographic terms: ", len(demographic_glove))
# demographic_glove_emb = np.array(demographic_glove_emb)
# demographic_glove_emb_mean = np.mean(demographic_glove_emb, axis=0)
# glove_demographic_dict['core'] = demographic_glove_emb_mean


# relational_glove = []
# relational_glove_emb = []

# for word in relational.vocab:
#     try:
#         emb = embedding_dict[word]
#         relational_glove.append(word)
#         relational_glove_emb.append(emb)
#     except:
#         pass

# print("Total # of relational terms: ", len(relational_glove))
# relational_glove_emb = np.array(relational_glove_emb)
# relational_glove_emb_mean = np.mean(relational_glove_emb, axis=0)
# glove_relational_dict['core'] = relational_glove_emb_mean


######Creating Nested Dictionary of Embedding Mean Values ######

word2vec_dict['culture'] = word2vec_culture_dict
word2vec_dict['demographic'] = word2vec_demographic_dict
word2vec_dict['relational'] = word2vec_relational_dict

# doc2vec_dict['culture'] = doc2vec_culture_dict
# doc2vec_dict['demographic'] = doc2vec_demographic_dict
# doc2vec_dict['relational'] = doc2vec_relational_dict

# glove_dict['culture'] = glove_culture_dict
# glove_dict['demographic'] = glove_demographic_dict
# glove_dict['relational'] = glove_relational_dict

dictionary['word2vec'] = word2vec_dict
# dictionary['doc2vec'] = doc2vec_dict
# dictionary['glove'] = glove_dict



#writing function for common cosine similarity
def doc_words_cosine(doc_tag, dict_type, model_type, vocab_size):
    '''Returns a cosine similarity value between a document/school and a list of words of choice.
        
    Input: The model to be used, document tag, dictionary type (culture, demographic, relational),
    the model type (doc2vec, word2vec, or GloVe), and the vocab size for the dictionaries (core, etc).
    InferSent doesn't require this step.
            
    Output: A scalar cosine similarity value.
        '''
    word_vec_avg = dictionary[model_type][dict_type][vocab_size]
    i = 0 #testing
    if model_type == "doc2vec":
        doc_vec = model_dmm.docvecs[str(doc_tag)]
    elif model_type == 'word2vec':
        i += 1
        if (i % 1000 == 0):
            print(i)
        s_sentences = str.split(df[df.edited_filename == str(doc_tag)].text.item())
        #init list for word embs
        ls = []
        for word in s_sentences:
            try:
                ls.append(model_word2vec[word])
            except:
                pass
        doc_vec = np.mean(np.array(ls), axis=0)
    elif model_type == 'glove':
        s_sentences = str.split(df[df.edited_filename == str(doc_tag)].text.item())
        #init list for word embs
        ls = []
        for word in s_sentences:
            try:
                ls.append(embedding_dict[word])
            except:
                pass
        doc_vec = np.mean(np.array(ls), axis=0)
    else:
        print("Model Type unspecified.")
    return absolute(dot(doc_vec, word_vec_avg)/(norm(doc_vec)*norm(word_vec_avg)))

###### Calculating Cosine Similary Values ######

# culture_doc2vec_cosine = []
# for i in tqdm(range(len(df.text))):
#     try:
#         cosine_value = doc_words_cosine(df.iloc[i,].edited_filename, 'culture', 'doc2vec', 'core')
#     except:
#         print("document vector does not exist. inserting NAN.")
# #         cosine_value = NaN --commenting out to speed up the process.
#         cosine_value = 0
#     culture_doc2vec_cosine.append(cosine_value)
    
# demographic_doc2vec_cosine = []
# for i in tqdm(range(len(df.text))):
#     try:
#         cosine_value = doc_words_cosine(df.iloc[i,].edited_filename, 'demographic', 'doc2vec', 'core')
#     except:
#         print("document vector does not exist. inserting NAN.")
# #         cosine_value = NaN
#         cosine_value = 0
#     demographic_doc2vec_cosine.append(cosine_value)
    
# relational_doc2vec_cosine = []
# for i in tqdm(range(len(df.text))):
#     try:
#         cosine_value = doc_words_cosine(df.iloc[i,].edited_filename, 'relational', 'doc2vec', 'core')
#     except:
#         print("document vector does not exist. inserting Nan.") 
#     relational_doc2vec_cosine.append(cosine_value)
    
# df['relational_doc2vec_cosine'] = relational_doc2vec_cosine
# df['demographic_doc2vec_cosine'] = demographic_doc2vec_cosine
# df['culture_doc2vec_cosine'] = culture_doc2vec_cosine


culture_word2vec_cosine = []
for i in tqdm(range(len(df.text))):
    cosine_value = doc_words_cosine(df.iloc[i,].edited_filename, 'culture', 'word2vec', 'core')
    culture_word2vec_cosine.append(cosine_value)
    
demographic_word2vec_cosine = []
for i in tqdm(range(len(df.text))):
    cosine_value = doc_words_cosine(df.iloc[i,].edited_filename, 'demographic', 'word2vec', 'core')
    demographic_word2vec_cosine.append(cosine_value)
    
relational_word2vec_cosine = []
for i in tqdm(range(len(df.text))):
    cosine_value = doc_words_cosine(df.iloc[i,].edited_filename, 'relational', 'word2vec', 'core')
    relational_word2vec_cosine.append(cosine_value)
    
df['relational_word2vec_cosine'] = relational_word2vec_cosine
df['demographic_word2vec_cosine'] = demographic_word2vec_cosine
df['culture_word2vec_cosine'] = culture_word2vec_cosine


# culture_glove_cosine = []
# for i in tqdm(range(len(df.text))):
#     cosine_value = doc_words_cosine(df.iloc[i,].edited_filename, 'culture', 'glove', 'core')
#     culture_glove_cosine.append(cosine_value)
    
# demographic_glove_cosine = []
# for i in tqdm(range(len(df.text))):
#     cosine_value = doc_words_cosine(df.iloc[i,].edited_filename, 'demographic', 'glove', 'core')
#     demographic_glove_cosine.append(cosine_value)
    
# relational_glove_cosine = []
# for i in tqdm(range(len(df.text))):
#     cosine_value = doc_words_cosine(df.iloc[i,].edited_filename, 'relational', 'glove', 'core')
#     relational_glove_cosine.append(cosine_value)
    
# df['relational_glove_cosine'] = relational_glove_cosine
# df['demographic_glove_cosine'] = demographic_glove_cosine
# df['culture_glove_cosine'] = culture_glove_cosine


df.to_csv("../../../models_storage/word_embeddings_data/text_with_cosine_scores_wdg_2020_oct17_1970.csv")

