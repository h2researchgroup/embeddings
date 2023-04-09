'''

@title: Similarity calculations for expanded dictionaries 
@author: Nancy Xu, UC Berkeley
@coauthors:  Jaren Haber, PhD, Dartmouth College; Prof. Heather Haveman, UC Berkeley
@contact: yinuoxu54@berkeley.edu

@project: Computational Literature Review of Organizational Scholarship
@repo: https://github.com/h2researchgroup/embeddings/
@date: April 5, 2023

------------------------------------------------------------------------------------------------

@description: Output csv with the 500 expanded terms (most cosine similar words from words in the core dictionaries) or refined terms (50 terms) per dict/time period, with each word's cosine distance from that dict, that word's overall frequency, and a tuple including (a) each of the 3 closest terms in seed dictionary and (b) the cosine distance from that term to the focal term.

------------------------------------------------------------------------------------------------

@script inputs: 
- work_dir (-w) --> the current directory you're working with, it should be one level higher than models_storage and dictionary_methods
- output_dir (-o) --> the directory where you want to store the csv dataframes
- get_expanded_or_refined (-e) --> output csvs with similar terms for expanded dictionaries (500 terms) or refined dictionaries (50 terms) 
    - 1 = expanded (500 closest terms to core dictionaries)
    - 2 = refined (50-term, hand-cleaned dicts)
    - 3 = both

@script outputs:
- stored in output_dir, csv dataframes containing cosine similarities of expanded dicts (500) for each decade and each perspective (12 total)
- stored in output_dir, csv dataframes containing cosine similarities of refined dicts for each decade and each perspective (12 total) 


@usage: python3 get_closest_words.py -w <path-to-work-dir> -o <path-to-output-dir> -e <get-expanded-or-refined> 

------------------------------------------------------------------------------------------------

@files used (see specific file paths in the "define filepaths section"):
- decade w2v models
- core dictionaries
- expanded decade dictionaries


'''

import gensim
from gensim.models import KeyedVectors
import pandas as pd
import sklearn
import sklearn.metrics
import numpy as np
import os
import argparse
from os.path import join


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output_dir", help = "Show Output" )
parser.add_argument("-w", "--work_dir", help = "Show Output" )
parser.add_argument("-e", "--get_expanded_or_refined", help = "Show Output" )


args = parser.parse_args()

work_dir = str(args.work_dir)
output_dir = str(args.output_dir)

get_expanded_or_refined = int(args.get_expanded_or_refined)


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
ma = KeyedVectors.load(w2v_all_decades_fp)


####################################
###### load in core dictionaries
####################################

cult_core = list(pd.read_csv(cult_core_fp, header=None)[0].apply(lambda x: x.replace(' ', '_')))
dem_core = list(pd.read_csv(dem_core_fp, header=None)[0].apply(lambda x: x.replace(' ', '_')))
relt_core = list(pd.read_csv(relt_core_fp, header=None)[0].apply(lambda x: x.replace(' ', '_')))


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

def remove_terms(seed_lst, m):
    """
    Remove terms from list of seed words that don't exist in model's vocab
    
    inputs:
        - seed_lst (list of str) = list of seed terms
        - m = a word2vec model
    
    outputs
        - new list of words
    
    """
    for i in range(5):
        for word in seed_lst:
            try: 
                if m.wv.key_to_index[word] > 0:
                    pass
            except:
                    seed_lst.remove(word)
                    print("Removed " + str(word) + " from  dictionary.")
        for word in seed_lst:
            try: 
                if m.wv.key_to_index[word] > 0:
                    pass
            except:
                    seed_lst.remove(word)
                    print("Removed " + str(word) + " from  dictionary.")


    print("Length of culture dictionary filtered into vector space:", len(seed_lst))
    return seed_lst


def get_cos_sim_w_core(expanded_words,core_words, model):
  """
  Using w2v model, get the avg cosine similarity scores and most similar terms
  for each term in expanded_words with the words in the core dictionary
  
  inputs:
      - expanded_words (list of str)= a list of words to compare with core dictionary words
      - core_words (list of str) = a list of core dictionary words
      - model = w2v model
  
  outputs:
      - avg_cos_sims (list of floats) = a list of cosine similaries denoting the average cosine similarity between each term in expanded_words
          with all the terms in core_words
      - top3_tuple_lst (list of tuples)= a list of tuples (term, cosine similarity) of the top 3 terms that are most similar to each word in 
          expanded_words
  
  """
  avg_cos_sims=[]
  top3_tuple_lst = []
  for w1 in expanded_words:
    try:
        ### get cosine similarity of w1 with all the core words
        sims_with_core_terms = sklearn.metrics.pairwise.cosine_similarity([model.wv[w1]],model.wv[core_words])[0]
        ## average
        avg_cos_sims.append(np.mean(sims_with_core_terms))
        ### get top 3 most similar core words to w1
        top3_df = pd.DataFrame(sims_with_core_terms,core_words ).reset_index().sort_values(0, ascending = False).head(3)
        top3_tuples = list(top3_df.itertuples(index=False, name=None))
        top3_tuple_lst.append(top3_tuples)
    except:
        avg_cos_sims.append('skip')
        top3_tuple_lst.append('skip')
        print(w1)
  return avg_cos_sims, top3_tuple_lst


def get_output_df(top_500_all, core, m ):
    
  """
  Get dataframe containing all the words in the list top_500_all, their average cosine similarity with the core,
  and the closest 3 terms from the core
  
  inputs:
      - top_500_all (list of str)= a list of words to compare with the core words
      - core (list of str)= a list of core seed terms
      - m = w2v model
  
  outputs:
      - df_concat (pandas df) = dataframe with the columns 'term', 'cosine_sim_with_core', ' closest_terms_from_core'
  
  
  """
    
  cos_avg_sims_top_500_all, top3_tuple_lst_top_500_all = get_cos_sim_w_core(top_500_all,core,m )
  cos_avg_sims_core_all, top3_tuple_lst_core_all= get_cos_sim_w_core(core,core,m )
  core_df = pd.DataFrame([core, cos_avg_sims_core_all,  top3_tuple_lst_core_all]).T
  top_500_df = pd.DataFrame([top_500_all, cos_avg_sims_top_500_all,  top3_tuple_lst_top_500_all]).T
  df_concat = pd.concat([core_df, top_500_df])
  df_concat.columns = ['term', 'cosine_sim_with_core', ' closest_terms_from_core']
  return df_concat

def get_results_decade(model, cult_core, dem_core, relt_core, cult_dict, dem_dict, relt_dict, get_500=False):
    
    """
    For each perspective, get dataframe containing all the words in the list top_500_all, their average cosine similarity with the core dict and the closest 3 terms from the core dict
    
    input:
        - model = w2v model
        - cult_core(list of str) = list of core words of cultural perspective
        - dem_core (list of str)= list of core words of demographic perspective
        - relt_core (list of str)= list of core words of relational perspective
        - cult_dict (list of str)= list of refined terms for cultural perspective, 
            could be empty list when getting similarities of expanded dicts 
            (500 terms closest core terms)
        - dem_dict (list of str)= list of refined terms for demographic perspective, 
            could be empty list when getting similarities of expanded dicts 
            (500 terms closest core terms)   
        - relt_dict (list of str)= list of refined terms for relational perspective, 
            could be empty list when getting similarities of expanded dicts 
            (500 terms closest core terms)   
        - get_500 (bool)= boolean indicating whether to use generated 500 expanded similar terms instead of using refined terms
   
   outputs:
       - cult_concat (pandas df) = cultural dataframe with the columns 'term', 'cosine_sim_with_core', ' closest_terms_from_core'
       - dem_concat (pandas df)= demographic dataframe with the columns 'term', 'cosine_sim_with_core', ' closest_terms_from_core'
       - relt_concat (pandas df) = relational dataframe with the columns 'term', 'cosine_sim_with_core', ' closest_terms_from_core'
    
    """
    # remove terms from each dictionary not in w2v model vocab
    
    dem_core=remove_terms(dem_core, model)
    cult_core=remove_terms(cult_core, model)
    relt_core=remove_terms(relt_core, model)
    
    if get_500:
        cult_dict = [i[0] for i in model.wv.most_similar(cult_core, topn = 500)]
        dem_dict = [i[0] for i in model.wv.most_similar(dem_core, topn = 500)]
        relt_dict = [i[0] for i in model.wv.most_similar(relt_core, topn = 500)]
    
    cult_concat=get_output_df(cult_dict, cult_core,model)
    dem_concat=get_output_df(dem_dict, dem_core,model)
    relt_concat=get_output_df(relt_dict, relt_core,model)
    return cult_concat, dem_concat, relt_concat



def get_results_save_expanded(output_dir):
    '''
    save csv's for the similarity metrics of expanded (500) terms
    '''
    # run similarities function over all three dicts for each decade
    
    cult_exp1, dem_exp1, relt_exp1=get_results_decade(m1, cult_core, dem_core, relt_core, [], [], [], get_500=True)

    cult_exp2, dem_exp2, relt_exp2=get_results_decade(m2, cult_core, dem_core, relt_core, [], [], [], get_500=True)

    cult_exp3, dem_exp3, relt_exp3=get_results_decade(m3, cult_core, dem_core, relt_core, [], [], [], get_500=True)

    cult_exp4, dem_exp4, relt_exp4=get_results_decade(m4, cult_core, dem_core, relt_core, [], [], [], get_500=True)

    cult_exp1.to_csv(join(output_dir,"/cos_sim_df_expanded_500_cult_1971-1981.csv"))
    dem_exp1.to_csv(output_dir+"/cos_sim_df_expanded_500_dem_1971-1981.csv")
    relt_exp1.to_csv(output_dir+"/cos_sim_df_expanded_500_relt_1971-1981.csv")

    cult_exp2.to_csv(output_dir+"/cos_sim_df_expanded_500_cult_1982_1992.csv")
    dem_exp2.to_csv(output_dir+"/cos_sim_df_expanded_500_dem_1982_1992.csv")
    relt_exp2.to_csv(output_dir+"/cos_sim_df_expanded_500_relt_1982_1992.csv")

    cult_exp3.to_csv(output_dir+"/cos_sim_df_expanded_500_cult_1993_2003.csv")
    dem_exp3.to_csv(output_dir+"/cos_sim_df_expanded_500_dem_1993_2003.csv")
    relt_exp3.to_csv(output_dir+"/cos_sim_df_expanded_500_relt_1993_2003.csv")

    cult_exp4.to_csv(output_dir+"/cos_sim_df_expanded_500_cult_2004_2014.csv")
    dem_exp4.to_csv(output_dir+"/cos_sim_df_expanded_500_dem_2004_2014.csv")
    relt_exp4.to_csv(output_dir+"/cos_sim_df_expanded_500_relt_2004_2014.csv")


def get_results_save_refined(output_dir):
    '''
    save csv's for the similarity metrics of refined (50) terms
    '''
    # run similarities function over all three dicts for each decade
    cult_ref1, dem_ref1, relt_ref1=get_results_decade(m1, cult_core, dem_core, relt_core, cult_refined_1, dem_refined_1, relt_refined_1)

    cult_ref2, dem_ref2, relt_ref2=get_results_decade(m2, cult_core, dem_core, relt_core, cult_refined_2, dem_refined_2, relt_refined_2)

    cult_ref3, dem_ref3, relt_ref3=get_results_decade(m3, cult_core, dem_core, relt_core, cult_refined_3, dem_refined_3, relt_refined_3)

    cult_ref4, dem_ref4, relt_ref4=get_results_decade(m4, cult_core, dem_core, relt_core, cult_refined_4, dem_refined_4, relt_refined_4)

    # Save similarities to disk in CSV format
    cult_ref1.to_csv(output_dir+"/cos_sim_df_refined_cult_1971-1981.csv")
    dem_ref1.to_csv(output_dir+"/cos_sim_df_refined_dem_1971-1981.csv")
    relt_ref1.to_csv(output_dir+"/cos_sim_df_refined_relt_1971-1981.csv")

    cult_ref2.to_csv(output_dir+"/cos_sim_df_refined_cult_1982_1992.csv")
    dem_ref2.to_csv(output_dir+"/cos_sim_df_refined_dem_1982_1992.csv")
    relt_ref2.to_csv(output_dir+"/cos_sim_df_refined_relt_1982_1992.csv")

    cult_ref3.to_csv(output_dir+"/cos_sim_df_refined_cult_1993_2003.csv")
    dem_ref3.to_csv(output_dir+"/cos_sim_df_refined_dem_1993_2003.csv")
    relt_ref3.to_csv(output_dir+"/cos_sim_df_refined_relt_1993_2003.csv")

    cult_ref4.to_csv(output_dir+"/cos_sim_df_refined_cult_2004_2014.csv")
    dem_ref4.to_csv(output_dir+"/cos_sim_df_refined_dem_2004_2014.csv")
    relt_ref4.to_csv(output_dir+"/cos_sim_df_refined_relt_2004_2014.csv")
     
if get_expanded_or_refined == 1:
    ### only get df for expanded (500) dicts
    get_results_save_expanded(output_dir)
elif get_expanded_or_refined == 2:
     ### only get df for refined (50) dicts
    get_results_save_refined(output_dir)
else:
    ### both
    get_results_save_expanded(output_dir)
    get_results_save_refined(output_dir)