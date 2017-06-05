#!/usr/bin/env bash

train_load_map = {
    'lengths'                 :          load_train_lengths,
    'common_words'            :          load_train_common_words,
    'metrics'                :          load__train_metrics,
    'tfidf_new'               :          load_train_tfidf_new,
    'magic'                   :          load_train_magic,
    'wh'                      :          load_wh_train,
    'one_upper'               :          load_one_upper_train,
    'topNs_avg_tok_freq'      :          load_topNs_avg_tok_freq_train,
    'abi'                     :          load_abi_train,
    'max_k_cores'             :          load_max_k_cores_train,
    'word2vec_metrics'        :          load_word2vec_metrics_train,
    'glove_metrics'           :          load_glove_metrics_train,
    'lex_metrics'             :          load_lex_metrics_train,
    'aux_pairs_50'            :          load_aux_pairs_50_train
}

names, n_estimators, subsample, colsample, max_depth




nr generic_stacking_template.py lengths,common_words 500 0.8 0.8 5