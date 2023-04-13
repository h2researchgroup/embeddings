'''
@title: Calculate coherence & distinctiveness scores for dictionaries by decade

@authors: Nancy Xu, UC Berkeley; Jaren Haber, PhD, Dartmouth College

@PI: Prof. Heather Haveman, UC Berkeley

@date: April 13, 2023

@description: This notebook calculates and visualizes the coherence score and distinctiveness score of core and refined dictionaries over decades. Code for calculation based on code here: https://github.com/jhaber-zz/charters4textxd2018/blob/master/notebooks/wem_hackathon_TextXD18.ipynb

------------------------------------------------------------------------------------------------

@script inputs: 
- work_dir (-w) --> the current directory you're working with, it should be one level higher than models_storage and dictionary_methods
- output_dir (-o) --> the directory where you want to store the csv dataframes

@script outputs:
- graph of coherence scores over decades for core dictionaries
- graph of coherence scores over decades for refined dictionaries
- graph of distinctiveness scores over decades for core dictionaries
- graph of distinctiveness scores over decades for refined dictionaries
- csv of coherence scores over decades for core dictionaries
- csv of coherence scores over decades for expanded dictionaries
- csv of distinctiveness scores over decades for core dictionaries
- csv of distinctiveness scores over decades for expanded dictionaries

'''

import numpy as np
import pandas as pd
import gensim
import re
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import os
import argparse
from os.path import join
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output_dir", help = "Show Output" )
parser.add_argument("-w", "--work_dir", help = "Show Output" )


args = parser.parse_args()

work_dir = str(args.work_dir)
output_dir = str(args.output_dir)

####################################
###### define filepaths
####################################

## models
models_dir = join(work_dir, 'models_storage/w2v_models')
w2v_1970s_fp = join(models_dir, 'word2vec_1971-1981_phrased_filtered_enchant_orgdict_300d_10w_020423.bin')
w2v_1980s_fp = join(models_dir, 'word2vec_1982-1992_phrased_filtered_enchant_orgdict_300d_10w_020423.bin')
w2v_1990s_fp = join(models_dir, 'word2vec_1993-2003_phrased_filtered_enchant_orgdict_300d_10w_020423.bin')
w2v_2000s_fp = join(models_dir, 'word2vec_2004-2014_phrased_filtered_enchant_orgdict_300d_10w_020423.bin')
w2v_all_decades_fp = join(models_dir,'word2vec_ALLYEARS_phrased_filtered_enchant_orgdict_300d_10w_020423.bin') 


## dicts
dicts_dir = join(work_dir, 'dictionary_methods/dictionaries')


## core dicts
core_dicts_dir = join(dicts_dir, 'core')
cult_core_fp = join(core_dicts_dir, 'cultural_core.csv')
dem_core_fp = join(core_dicts_dir, 'demographic_core.csv')
relt_core_fp = join(core_dicts_dir, 'relational_core.csv')


## refined dicts
refined_dicts_dir = join(dicts_dir, 'expanded_decades')

cult_refined_decade_1_fp = join(refined_dicts_dir, 'cultural_1971_1981.txt')
dem_refined_decade_1_fp = join(refined_dicts_dir, 'demographic_1971_1981.txt')
relt_refined_decade_1_fp = join(refined_dicts_dir, 'relational_1971_1981.txt')

cult_refined_decade_2_fp = join(refined_dicts_dir, 'cultural_1982_1992.txt')
dem_refined_decade_2_fp = join(refined_dicts_dir, 'demographic_1982_1992.txt')
relt_refined_decade_2_fp = join(refined_dicts_dir, 'relational_1982_1992.txt')

cult_refined_decade_3_fp = join(refined_dicts_dir, 'cultural_1993_2003.txt')
dem_refined_decade_3_fp = join(refined_dicts_dir, 'demographic_1993_2003.txt')
relt_refined_decade_3_fp = join(refined_dicts_dir, 'relational_1993_2003.txt')

cult_refined_decade_4_fp = join(refined_dicts_dir, 'cultural_2004_2014.txt')
dem_refined_decade_4_fp = join(refined_dicts_dir, 'demographic_2004_2014.txt')
relt_refined_decade_4_fp = join(refined_dicts_dir, 'relational_2004_2014.txt')


####################################
###### load in decade specific models
####################################

m1 = KeyedVectors.load(w2v_1970s_fp)
m2 = KeyedVectors.load(w2v_1980s_fp)
m3 = KeyedVectors.load(w2v_1990s_fp)
m4 = KeyedVectors.load(w2v_2000s_fp)

models = [m1, m2, m3, m4]

####################################
###### load in core dictionaries
####################################

cult_core = list(pd.read_csv(cult_core_fp, header=None)[0].apply(lambda x: x.replace(' ', '_')))
dem_core = list(pd.read_csv(dem_core_fp, header=None)[0].apply(lambda x: x.replace(' ', '_')))
relt_core = list(pd.read_csv(relt_core_fp, header=None)[0].apply(lambda x: x.replace(' ', '_')))

core_lists = [dem_core,relt_core,cult_core]
years = ['1971-1981','1982-1992','1993-2003','2004-2014']

####################################
###### load in expanded decade dictionaries
####################################

cult_refined_1 = list(pd.read_csv(cult_refined_decade_1_fp, header=None)[0].apply(lambda x: x.replace(' ', '_')))
dem_refined_1 = list(pd.read_csv(dem_refined_decade_1_fp, header=None)[0].apply(lambda x: x.replace(' ', '_')))
relt_refined_1 = list(pd.read_csv(relt_refined_decade_1_fp, header=None)[0].apply(lambda x: x.replace(' ', '_')))

cult_refined_2 = list(pd.read_csv(cult_refined_decade_2_fp, header=None)[0].apply(lambda x: x.replace(' ', '_')))
dem_refined_2 = list(pd.read_csv(dem_refined_decade_2_fp, header=None)[0].apply(lambda x: x.replace(' ', '_')))
relt_refined_2 = list(pd.read_csv(relt_refined_decade_2_fp, header=None)[0].apply(lambda x: x.replace(' ', '_')))

cult_refined_3 = list(pd.read_csv(cult_refined_decade_3_fp, header=None)[0].apply(lambda x: x.replace(' ', '_')))
dem_refined_3 = list(pd.read_csv(dem_refined_decade_3_fp, header=None)[0].apply(lambda x: x.replace(' ', '_')))
relt_refined_3 = list(pd.read_csv(relt_refined_decade_3_fp, header=None)[0].apply(lambda x: x.replace(' ', '_')))

cult_refined_4 = list(pd.read_csv(cult_refined_decade_4_fp, header=None)[0].apply(lambda x: x.replace(' ', '_')))
dem_refined_4 = list(pd.read_csv(dem_refined_decade_4_fp, header=None)[0].apply(lambda x: x.replace(' ', '_')))
relt_refined_4 = list(pd.read_csv(relt_refined_decade_4_fp, header=None)[0].apply(lambda x: x.replace(' ', '_')))

####################################
###### define functions
####################################

def dict_cohere(thisdict, wem_model):
    '''Computes the average cosine similarity score of terms within one dictionary with all other terms in that same dictionary,
    effectively measuring the coherence of the dictionary.
    ...question for development: does it make sense to compare the average cosine similarity score between all terms 
    in thisdict and the average cosine similarity among the total model vocabulary? (Could that be, by definition, 0?)
    
    NOTE: For an unknown reason, calling this function deletes terms from thisdict.
    
    Inputs: List of key terms, word2vec model.
    Output: Average cosine similarity score of each word with all other words in the list of key terms.'''
    
    # Initialize average distance variables:

    sim_scores = []
    
    # Compute average cosine similarity score of each word with other dict words:
    for index in np.arange(len(thisdict)):
 
        other_words = thisdict[:index] + thisdict[index+1:]
        word = thisdict[index]
        sim_score_with_others = []
        for other in other_words:
          try:
            sim_score_with_others.append(wem_model.wv.similarity(word, other))
          except:
            pass
        
        if (len(sim_score_with_others)!=0):
          word_avg_sim = np.mean(sim_score_with_others)
          # print(word_avg_sim)
          sim_scores.append(word_avg_sim)# Add up each average distance, incrementally
     

    
    return np.mean(sim_scores)

## Calculate distinctiveness scores of each dictionary

def dict_distinct(dict1, dict2, wem_model):
    '''Computes the average cosine distance score of terms in dict1 with all terms in dict2,
    effectively measuring the opposition/non-coherence between the two dictionaries.
    
    NOTE: For an unknown reason, calling this function deletes terms from thisdict.
    
    Inputs: List of key terms, word2vec model.
    Output: Average cosine distance score of each word in dict1 with all words in dict2.'''
    
    
    sim_scores = []
    
    # Compute average cosine distance score of each word with other dict words:
    for index in np.arange(len(dict1)):
 
        other_words = dict2
        word = dict1[index]
        sim_score_with_others = []
        for other in other_words:
          try:
            ## get cosine distance 
            sim_score_with_others.append(1-wem_model.wv.similarity(word, other))
          except:
            pass
        
        if (len(sim_score_with_others)!=0):
          word_avg_sim = np.mean(sim_score_with_others)
          sim_scores.append(word_avg_sim)# Add up each average distance, incrementally
     

    
    return np.mean(sim_scores)

def get_distinct_score_df(l1=None, l2=None, l3=None, l4=None, expanded = False):
    """
    
    get a pandas dataframe that contains the distinctiveness score for each decade and each dictionary
    
    input:
        
        l1 = list of expanded dictionaries for decade 1 (None for core decade dicts)
        l2 = list of expanded dictionaries for decade 2 (None for core decade dicts)
        l3 = list of expanded dictionaries for decade 3 (None for core decade dicts)
        l4 = list of expanded dictionaries for decade 4 (None for core decade dicts)
        expanded = boolean indicating whether we get the df for  expanded decade dicts  or core decade dicts
    
    output: 
        df storing distinctive score for each decade dicts 
    
    
    """
    ## if we want to get the scores for expanded decade dicts, 
    ## using the expanded dictionaries passed as inputs 
    if expanded: 
        year_coresp_dict_lst={}
        year_coresp_dict_lst['1971-1981'] = l1
        year_coresp_dict_lst['1982-1992'] = l2
        year_coresp_dict_lst['1993-2003'] = l3
        year_coresp_dict_lst['2004-2014'] = l4

    
    distinct_scores_decade_perspective={}
   
    ### iterate through each of the 4 periods, using the period-specific models
    
    for y,m in zip(years, models):
        
        distinct_scores_perspective={}
        
        ### for each period, iterate through the 3 perspectives to calculate their distinct scores
        for core,k in zip(['cult','dem','relt'],np.arange(3)):
            
            if expanded:
                ### use the expanded decade dicts
                current_lst = year_coresp_dict_lst[y][k]
                other_lsts = year_coresp_dict_lst[y][:k] + year_coresp_dict_lst[y][k+1:]
            else:
                ### use the core dicts
                current_lst = core_lists[k]
                other_lsts = core_lists[:k] + core_lists[k+1:]
           
            ## get the the mean distinct score for this dictionary with the dicts of the other 2 perspectives
            distinct_scores_perspective[core]= np.mean([dict_distinct(current_lst, other_lsts[0], m), dict_distinct(current_lst, other_lsts[1], m)])
        
        ## store the distrinct scores for each of the 3 perspectives for each year 
        distinct_scores_decade_perspective[y] = distinct_scores_perspective
        
    
    distinct_results= pd.DataFrame(distinct_scores_decade_perspective).T
    
    return distinct_results
    
    
## calculate coherence score of core decade dicts

dem_1970_cohere = dict_cohere(dem_core,m1)
cult_1970_cohere = dict_cohere(cult_core,m1)
rela_1970_cohere = dict_cohere(relt_core,m1)

dem_1980_cohere = dict_cohere(dem_core,m2)
cult_1980_cohere = dict_cohere(cult_core,m2)
rela_1980_cohere = dict_cohere(relt_core,m2)

dem_1990_cohere = dict_cohere(dem_core,m3)
cult_1990_cohere = dict_cohere(cult_core,m3)
rela_1990_cohere = dict_cohere(relt_core,m3)

dem_2000_cohere = dict_cohere(dem_core,m4)
cult_2000_cohere = dict_cohere(cult_core,m4)
rela_2000_cohere = dict_cohere(relt_core,m4)

## calculate coherence score of refined decade dicts
dem_1970_cohere_ref = dict_cohere(list(dem_refined_1),m1)
cult_1970_cohere_ref = dict_cohere(list(cult_refined_1),m1)
rela_1970_cohere_ref = dict_cohere(list(relt_refined_1),m1)

dem_1980_cohere_ref = dict_cohere(list(dem_refined_2),m2)
cult_1980_cohere_ref = dict_cohere(list(cult_refined_2),m2)
rela_1980_cohere_ref = dict_cohere(list(relt_refined_2),m2)

dem_1990_cohere_ref = dict_cohere(list(dem_refined_3),m3)
cult_1990_cohere_ref = dict_cohere(list(cult_refined_3),m3)
rela_1990_cohere_ref = dict_cohere(list(relt_refined_3),m3)

dem_2000_cohere_ref = dict_cohere(list(dem_refined_4),m4)
cult_2000_cohere_ref = dict_cohere(list(cult_refined_4),m4)
rela_2000_cohere_ref = dict_cohere(list(relt_refined_4),m4)


### save the results to csv's in the output directory

cohere_results_ref = pd.DataFrame([dem_1970_cohere_ref,dem_1980_cohere_ref,dem_1990_cohere_ref,dem_2000_cohere_ref])
cohere_results_ref['cult'] = [cult_1970_cohere_ref,cult_1980_cohere_ref,cult_1990_cohere_ref,cult_2000_cohere_ref]
cohere_results_ref['relt'] = [rela_1970_cohere_ref,rela_1980_cohere_ref,rela_1990_cohere_ref,rela_2000_cohere_ref]
cohere_results_ref.columns = ['dem', 'cult', 'relt']
cohere_results_ref=cohere_results_ref.set_axis(years)

cohere_results_ref.to_csv(join(output_dir, 'cohere_results_ref.csv'))

cohere_results = pd.DataFrame([dem_1970_cohere,dem_1980_cohere,dem_1990_cohere,dem_2000_cohere])
cohere_results['cult'] = [cult_1970_cohere,cult_1980_cohere,cult_1990_cohere,cult_2000_cohere]
cohere_results['relt'] = [rela_1970_cohere,rela_1980_cohere,rela_1990_cohere,rela_2000_cohere]
cohere_results.columns = ['dem', 'cult', 'relt']
cohere_results=cohere_results.set_axis(years)

cohere_results.to_csv(join(output_dir, 'cohere_results_core.csv'))


#graph coherence score for expanded decade dicts
plt.plot(years,[dem_1970_cohere_ref,dem_1980_cohere_ref,dem_1990_cohere_ref,dem_2000_cohere_ref],'--', label='Organizational Ecology', color = 'green')

plt.plot(years,[cult_1970_cohere_ref,cult_1980_cohere_ref,cult_1990_cohere_ref,cult_2000_cohere_ref],label='Organizational Institutionalism', color = 'red')

plt.plot(years,[rela_1970_cohere_ref,rela_1980_cohere_ref,rela_1990_cohere_ref,rela_2000_cohere_ref],':', label='Resource Dependence', color = 'blue')

plt.xlabel('Time period')
plt.ylabel('Coherence')

ax = plt.subplot(111)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=3, frameon=False)


plt.ylim(0.15, 0.35)
plt.yticks(np.linspace(0.15, 0.35, 5))
plt.grid()

filepath = join(output_dir, 'coherence_score_expanded_dict' + ".png")
plt.savefig(filepath, bbox_inches='tight', dpi= 2000)


# graph coherence score for core decade dicts

import matplotlib.pyplot as plt
plt.plot(years,[dem_1970_cohere,dem_1980_cohere,dem_1990_cohere,dem_2000_cohere],'--', label='Organizational Ecology', color = 'green')

plt.plot(years,[cult_1970_cohere,cult_1980_cohere,cult_1990_cohere,cult_2000_cohere], label='Organizational Institutionalism', color = 'red')

plt.plot(years,[rela_1970_cohere,rela_1980_cohere,rela_1990_cohere,rela_2000_cohere],':', label='Resource Dependence', color = 'blue')

plt.xlabel('Time period')
plt.ylabel('Coherence')

ax = plt.subplot(111)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=3, frameon=False)

plt.ylim(0.15, 0.35)
plt.yticks(np.linspace(0.15, 0.35, 5))
plt.grid()
filepath = join(output_dir,  'coherence_score_core_dict' + ".png")

plt.savefig(filepath, bbox_inches='tight', dpi= 2000)

    
    
### get a pandas df storing distinctive score for each refined decade dicts 
ref_list_1 =[cult_refined_1, dem_refined_1, relt_refined_1] 
ref_list_2 =[cult_refined_2, dem_refined_2, relt_refined_2] 
ref_list_3 =[cult_refined_3, dem_refined_3, relt_refined_3] 
ref_list_4 =[cult_refined_4, dem_refined_4, relt_refined_4] 
distinct_results_ref = get_distinct_score_df(ref_list_1,ref_list_2,ref_list_3, ref_list_4, True  )

distinct_results_ref.to_csv(join(output_dir, 'distinct_results_ref.csv'))
print("saved distinct_results_ref.csv!")

### get a pandas df storing distinctive score for each core decade dicts 
core_lists = [dem_core,relt_core,cult_core]
distinct_results = get_distinct_score_df(expanded =False  )

distinct_results.to_csv(join(output_dir, 'distinct_results_core.csv'))
print("saved distinct_results_core..csv!")

### plot distinctive score for expanded decade dicts
plt.plot(distinct_results_ref['dem'],'--', label='Organizational Ecology', color = 'green')

plt.plot(distinct_results_ref['relt'], label='Organizational Institutionalism', color = 'red')

plt.plot(distinct_results_ref['cult'],':', label='Resource Dependence', color = 'blue')

plt.xlabel('Time period')
plt.ylabel('Distinctiveness')
ax = plt.subplot(111)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=3, frameon=False)
# plt.title('Distinctiveness (cosine distance) vs. decades')

plt.ylim(0.75, 0.95)
plt.yticks(np.linspace(0.75,  0.95, 5))
plt.grid()

filepath = join(output_dir,'distinctiveness_score_refined_dict' + ".png")

plt.savefig(filepath, bbox_inches='tight', dpi= 2000)

### plot distinctive score for core decade dicts

plt.plot(distinct_results['dem'], '--', label='Organizational Ecology', color = 'green')

plt.plot(distinct_results['relt'],label='Organizational Institutionalism', color = 'red')

plt.plot(distinct_results['cult'], ':',label='Resource Dependence', color = 'blue')

plt.xlabel('Time period')
plt.ylabel('Distinctiveness')


ax = plt.subplot(111)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=3, frameon=False)


plt.ylim(0.75, 0.95)
plt.yticks(np.linspace(0.75,  0.95, 5))
plt.grid()


filepath = join( output_dir, 'distinctiveness_score_core_dict' + ".png")

plt.savefig(filepath, bbox_inches='tight', dpi= 2000)