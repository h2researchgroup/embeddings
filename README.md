# embeddings
This code takes in JSTOR OCR raw text and expert-generated dictionaries, computes embeddings, uses these to expand dictionaries, gets doc/dict cosine similarities, and visualizes overall trends.

## Guide to Codebase
#### Word2Vec:
*  script for training full and decade-specific word2vec models: <a href="word2vec/word2vec_train.py">word2vec/word2vec_train.py</a>
*  template notebook for exploring word2vec model running, with a full preprocessing workflow and decade-specific training: <a href="word2vec/w2v_nb_template_workflow.ipynb">word2vec/w2v_nb_template_workflow.ipynb</a>

#### Dictionary Method:
* script to use the w2v model to expand seed vocab:  <a href="refine_dictionaries/expand_dictonary_and_visualize.ipynb">refine_dictionaries/expand_dictonary_and_visualize.ipynb</a>
* script to use the w2v model to expand seed vocab for each decade and visualize TSNE:  <a href="refine_dictionaries/refine_dict.ipynb">refine_dictionaries/refine_dict.ipynb</a>

#### Validate and visualize trend:
* correlation:  <a href="validate/correlations_cosine_coredict_by_year.ipynb">validate/correlations_cosine_coredict_by_year.ipynb</a>
* trend of cosine score and ratio: <a href="validate/viz_patterns_coredict_ratios_cosines_normalized.ipynb">validate/viz_patterns_coredict_ratios_cosines_normalized.ipynb</a>
* trend of cosine score (updated publication year): <a href="validate/viz_trend_by_year.ipynb">validate/viz_trend_by_year.ipynb</a>

#### Clustering:
* Notebook to view hierarchical clusters based on seed dictionaries divided by decade:
    * <a href="clustering/hierarchical_by_decade.ipynb">clustering/hierarchical_by_decade.ipynb</a>
