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




nr generic_submit_stacking_template.py lengths,common_words 500 0.8 0.8 5 &
done submit
nr generic_submit_stacking_template.py metrics 500 0.8 0.8 5 &
done submit

nr generic_submit_stacking_template.py lengths,common_words,magic 500 0.8 0.8 5 &
done submit


nr generic_submit_stacking_template.py glove_metrics 500 0.8 0.8 5 &
done submit

nr generic_submit_stacking_template.py glove_metrics,tfidf_new 500 0.8 0.8 5 &
done submit
nr generic_submit_stacking_template.py glove_metrics,lex_metrics,word2vec_metrics 500 0.8 0.8 5 &
done submit


nr generic_submit_stacking_template.py topNs_avg_tok_freq 500 0.8 0.8 5 &
done submit

nr generic_submit_stacking_template.py topNs_avg_tok_freq,magic 500 0.8 0.8 5 &
done submit


nr generic_submit_stacking_template.py lengths,common_words,topNs_avg_tok_freq 500 0.6 0.6 5 &
done submit


nr generic_submit_stacking_template.py tfidf_new,magic 500 0.6 0.6 5 &
done submit

nr generic_submit_stacking_template.py tfidf_new,magic,word2vec_metrics 500 0.6 0.6 5 &
done submit

nr generic_submit_stacking_template.py tfidf_new,magic,topNs_avg_tok_freq 500 0.6 0.6 5 &
done submit


nr generic_submit_stacking_template.py one_upper,magic,wh,common_words,lengths 500 0.6 0.6 5 &
done submit