# embeddings
This code takes in JSTOR OCR raw text and expert-generated dictionaries, computes embeddings, uses these to expand dictionaries, gets doc/dict cosine similarities, and visualizes overall trends.

## Guide to Codebase
#### Word2Vec:
*  script for proprocessing texts and training w2v models: <a href="word2vec/word2vec.py">word2vec/word2vec.py</a>
*  script for preprocesses training texts, and creating decade-specific word2vec embeddings: <a href="word2vec/clean_text_train_decade_word2vec.ipynb">word2vec/clean_text_train_decade_word2vec.ipynb</a>

#### Dictionary Method:
* script to use the w2v model to expand seed vocab:  <a href="refine_dictionaries/expand_dictonary_and_visualize.ipynb">refine_dictionaries/expand_dictonary_and_visualize.ipynb</a>
* script to use the w2v model to expand seed vocab for each decade and visualize TSNE:  <a href="refine_dictionaries/refine_dict.ipynb">refine_dictionaries/refine_dict.ipynb</a>

#### Validate and visualize trend:
* correlation:  <a href="validate/correlations_cosine_coredict_by_year.ipynb">validate/correlations_cosine_coredict_by_year.ipynb</a>
* trend of cosine score and ratio: <a href="validate/viz_patterns_coredict_ratios_cosines_normalized.ipynb">validate/viz_patterns_coredict_ratios_cosines_normalized.ipynb</a>
* trend of cosine score (updated publication year): <a href="validate/viz_trend_by_year.ipynb">validate/viz_trend_by_year.ipynb</a>

