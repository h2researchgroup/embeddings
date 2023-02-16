#!/usr/bin/env python
# coding: utf-8

'''
@authors: Jaren Haber, PhD, Dartmouth College; Nancy Xu, UC Berkeley
@PI: Prof. Heather Haveman, UC Berkeley
@date_modified: February 2023
@contact: jhaber@berkeley.edu
@inputs: file with counts of each perspective per year list of authors OR list of terms per perspective, list of article filepaths 
@outputs: count of authors ((citations_by_persp|dicts)_count_{thisdate}.csv), where 'thisdate' is in mmddyy format 
@usage: run `python3 dict_count_all.py` from within `dictionary_methods/code`
@description: Counts mentions of perspectives--summed over individual words or authors--in articles by using ngram files ACROSS ALL YEARS OF DATA. To count a new list of terms or author names in the ngram files, change the list of words in the 'Update Words' section and/or change the word type to count (between 'citations' and 'terms', i.e. dictionaries), then run the script. 
'''

# Computational Literature Review
# 
# Creator: Jaren Haber, PhD, Nancy Xu
# 
# Date created: November 13, 2020
# 
# Date last modified: April 11, 2022
# 
# This notebook visualize the ratio and cosine trend over the years with corrected year data.

# In[1]:


###############################################
#                  Initialize                 #
###############################################

# Import packages
import pandas as pd
import numpy as np
from os.path import join
from os import getcwd
from datetime import date
import seaborn as sns
import matplotlib.pyplot as plt
fig = plt.gcf() # For saving better figures

# Define filepaths
cwd = getcwd()
root = str.replace(cwd, 'embeddings/validate', '')
FIGURES_HOME = join(root, 'embeddings/figures/engagement_plots')
DATA_HOME = join(root, 'dictionary_methods/article_data')
COUNTS = join(DATA_HOME, 'dicts_count_ALL_merged_021423.csv')


###############################################
#            Import & prepare data            #
###############################################

df = pd.read_csv(COUNTS, low_memory=False)
df['publicationYear'] = df['publicationYear'].astype(int)

print('Preparing data...')

# Standardize ratio cols
df['cultural_ratio'] = (df['cultural_ratio'] - df['cultural_ratio'].min())/(df['cultural_ratio'].max() - df['cultural_ratio'].min())
df['demographic_ratio'] = (df['demographic_ratio'] - df['demographic_ratio'].min())/(df['demographic_ratio'].max() - df['demographic_ratio'].min())
df['relational_ratio'] = (df['relational_ratio'] - df['relational_ratio'].min())/(df['relational_ratio'].max() - df['relational_ratio'].min())

# Get averages by year, splitting by discipline
keepcols = ['cultural_ratio', 'demographic_ratio', 'relational_ratio', 'publicationYear']
# Mean
df_avg_sociology = df[df['primary_subject'] == 'Sociology'][keepcols].groupby(by='publicationYear').mean().reset_index()
df_avg_management = df[df['primary_subject'] == 'Management & Organizational Behavior'][keepcols].groupby(by='publicationYear').mean().reset_index()

# 25th percentile
df_25_sociology = df[df['primary_subject'] == 'Sociology'][keepcols].groupby(by='publicationYear').quantile(0.25).reset_index()
df_25_management = df[df['primary_subject'] == 'Management & Organizational Behavior'][keepcols].groupby(by='publicationYear').quantile(0.25).reset_index()
df_25_sociology['percentile'] = '25th'
df_25_management['percentile'] = '25th'

# Median
df_50_sociology = df[df['primary_subject'] == 'Sociology'][keepcols].groupby(by='publicationYear').quantile(0.5).reset_index()
df_50_management = df[df['primary_subject'] == 'Management & Organizational Behavior'][keepcols].groupby(by='publicationYear').quantile(0.5).reset_index()
df_50_sociology['percentile'] = '50th'
df_50_management['percentile'] = '50th'

# 75th percentile
df_75_sociology = df[df['primary_subject'] == 'Sociology'][keepcols].groupby(by='publicationYear').quantile(0.75).reset_index()
df_75_management = df[df['primary_subject'] == 'Management & Organizational Behavior'][keepcols].groupby(by='publicationYear').quantile(0.75).reset_index()
df_75_sociology['percentile'] = '75th'
df_75_management['percentile'] = '75th'

# Join percentiles into one df per discipline
df_sociology = pd.concat([df_25_sociology, df_50_sociology, df_75_sociology]).reset_index()
df_management = pd.concat([df_25_management, df_50_management, df_75_management]).reset_index()


# Make long-format DFs for each discipline (for plotting purposes), using mean and median
# mean: soc
df_avg_long_soc = pd.melt(df_avg_sociology, id_vars=['publicationYear'])
df_avg_long_soc.columns = ['Year', 'Perspective', 'Engagement']
persp_conditions = [df_avg_long_soc.Perspective == 'cultural_ratio', df_avg_long_soc.Perspective == 'demographic_ratio', df_avg_long_soc.Perspective == 'relational_ratio']
persp_values = ['Cultural', 'Demographic', 'Relational']
df_avg_long_soc['Perspective'] = np.select(persp_conditions, persp_values)

# mean: mgmt
df_avg_long_mgmt = pd.melt(df_avg_management, id_vars=['publicationYear'])
df_avg_long_mgmt.columns = ['Year', 'Perspective', 'Engagement']
persp_conditions = [df_avg_long_mgmt.Perspective == 'cultural_ratio', df_avg_long_mgmt.Perspective == 'demographic_ratio', df_avg_long_mgmt.Perspective == 'relational_ratio']
df_avg_long_mgmt['Perspective'] = np.select(persp_conditions, persp_values)

# median: soc
df_50_long_soc = pd.melt(df_50_sociology.drop(columns=['percentile']), id_vars=['publicationYear'])
df_50_long_soc.columns = ['Year', 'Perspective', 'Engagement']
persp_conditions = [df_50_long_soc.Perspective == 'cultural_ratio', df_50_long_soc.Perspective == 'demographic_ratio', df_50_long_soc.Perspective == 'relational_ratio']
df_50_long_soc['Perspective'] = np.select(persp_conditions, persp_values)

# median: mgmt
df_50_long_mgmt = pd.melt(df_50_management.drop(columns=['percentile']), id_vars=['publicationYear'])
df_50_long_mgmt.columns = ['Year', 'Perspective', 'Engagement']
persp_conditions = [df_50_long_mgmt.Perspective == 'cultural_ratio', df_50_long_mgmt.Perspective == 'demographic_ratio', df_50_long_mgmt.Perspective == 'relational_ratio']
df_50_long_mgmt['Perspective'] = np.select(persp_conditions, persp_values)


###############################################
#      Plot engagement: all perspectives      #
###############################################

print('Visualizing all perspectives...')

# All perspectives: Sociology (mean)
ax = sns.lineplot(x="Year", y="Engagement", data=df_avg_long_soc, hue = 'Perspective')
ax.set(xlabel='Year', ylabel='Engagement')
ax.set(ylim=(0.0, 0.09))

ax.legend(loc='upper center', bbox_to_anchor=(1.3, 0.5), ncol=3)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3)

fig.set_size_inches(8, 6)
thisday = date.today().strftime("%m%d%y") # get current date
title = f"ratio_all_sociology_{thisday}"
plt.savefig(join(FIGURES_HOME, title + ".png"), dpi=1200, bbox_inches="tight")
plt.close()


# All perspectives: Management (mean)
ax = sns.lineplot(x="Year", y="Engagement", data=df_avg_long_mgmt, hue = 'Perspective')
ax.set(xlabel='Year', ylabel='Engagement')
ax.set(ylim=(0.0, 0.09))

ax.legend(loc='upper center', bbox_to_anchor=(1.3, 0.5), ncol=3)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3)

fig.set_size_inches(8, 6)
thisday = date.today().strftime("%m%d%y") # get current date
title = f"ratio_all_sociology_{thisday}"
plt.savefig(join(FIGURES_HOME, title + ".png"), dpi=1200, bbox_inches="tight")
plt.close()


# All perspectives: Sociology (median)
ax = sns.lineplot(x="Year", y="Engagement", data=df_50_long_soc, hue = 'Perspective')
ax.set(xlabel='Year', ylabel='Engagement')
ax.set(ylim=(0.0, 0.06))

ax.legend(loc='upper center', bbox_to_anchor=(1.3, 0.5), ncol=3)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3)

fig.set_size_inches(8, 6)
thisday = date.today().strftime("%m%d%y") # get current date
title = f"ratio_all_sociology_{thisday}"
plt.savefig(join(FIGURES_HOME, title + ".png"), dpi=1200, bbox_inches="tight")
plt.close()


# All perspectives: Management (median)
ax = sns.lineplot(x="Year", y="Engagement", data=df_50_long_mgmt, hue = 'Perspective')
ax.set(xlabel='Year', ylabel='Engagement')
ax.set(ylim=(0.0, 0.06))

ax.legend(loc='upper center', bbox_to_anchor=(1.3, 0.5), ncol=3)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3)

fig.set_size_inches(8, 6)
thisday = date.today().strftime("%m%d%y") # get current date
title = f"ratio_all_sociology_{thisday}"
plt.savefig(join(FIGURES_HOME, title + ".png"), dpi=1200, bbox_inches="tight")
plt.close()


###############################################
#   Plot engagement: 25/50/75th percentiles   #
###############################################

print('Visualizing percentiles...')

# Cultural: Sociology
ax = sns.lineplot(x="publicationYear", y="cultural_ratio", data=df_sociology, hue = 'percentile')
ax.set(xlabel='Year', ylabel='Engagement')
ax.set(ylim=(0.0, 0.11))

ax.legend(loc='upper center', bbox_to_anchor=(1.3, 0.5), ncol=3)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3)

fig.set_size_inches(8, 6)
thisday = date.today().strftime("%m%d%y") # get current date
title = f"ratio_cultural_sociology_{thisday}"
plt.savefig(join(FIGURES_HOME, title + ".png"), dpi=1200, bbox_inches="tight")
plt.close()


# Cultural: Management/OB
df_management = df_management[~df_management['publicationYear'].isin([1970,2016])]
ax = sns.lineplot(x="publicationYear", y="cultural_ratio", data=df_management, hue = 'percentile')
ax.set(xlabel='Year', ylabel='Engagement')
ax.set(ylim=(0.0, 0.11))

ax.legend(loc='upper center', bbox_to_anchor=(1.3, 0.5), ncol=3)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3)

fig.set_size_inches(8, 6)
thisday = date.today().strftime("%m%d%y") # get current date
title = f"ratio_cultural_mgmt_{thisday}"
plt.savefig(join(FIGURES_HOME, title + ".png"), dpi=1200, bbox_inches="tight")
plt.close()


# Demographic: Sociology
ax = sns.lineplot(x="publicationYear", y="demographic_ratio", data=df_sociology, hue = 'percentile')
ax.set(xlabel='Year', ylabel='Engagement')
ax.set(ylim=(0.0, 0.11))

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3)

fig.set_size_inches(8, 6)
thisday = date.today().strftime("%m%d%y") # get current date
title = f"ratio_demographic_sociology_{thisday}"
plt.savefig(join(FIGURES_HOME, title + ".png"), dpi=200, bbox_inches="tight")
plt.close()


# Demographic: Management
ax = sns.lineplot(x="publicationYear", y="demographic_ratio", data=df_management, hue = 'percentile')
ax.set(xlabel='Year', ylabel='Engagement')
ax.set(ylim=(0.0, 0.11))

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3)

fig.set_size_inches(8, 6)
thisday = date.today().strftime("%m%d%y") # get current date
title = f"ratio_demographic_management_{thisday}"
plt.savefig(join(FIGURES_HOME, title + ".png"), dpi=200, bbox_inches="tight")
plt.close()


# Relational: Sociology
ax = sns.lineplot(x="publicationYear", y="relational_ratio", data=df_sociology, hue = 'percentile')
ax.set(xlabel='Year', ylabel='Engagement')
ax.set(ylim=(0.0, 0.11))

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3)

fig.set_size_inches(8, 6)
thisday = date.today().strftime("%m%d%y") # get current date
title = f"ratio_relational_sociology_{thisday}"
plt.savefig(join(FIGURES_HOME, title + ".png"), dpi=200, bbox_inches="tight")
plt.close()


# Relational: Management
ax = sns.lineplot(x="publicationYear", y="relational_ratio", data=df_management, hue = 'percentile')
ax.set(xlabel='Year', ylabel='Engagement')
ax.set(ylim=(0.0, 0.11))

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3)

fig.set_size_inches(8, 6)
thisday = date.today().strftime("%m%d%y") # get current date
title = f"ratio_relational_management_{thisday}"
plt.savefig(join(FIGURES_HOME, title + ".png"), dpi=200, bbox_inches="tight")
plt.close()
