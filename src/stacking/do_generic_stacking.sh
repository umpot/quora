#!/usr/bin/env bash






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
    'aux_pairs_50'        :    load_aux_pairs_50_train,
    'pronoun_pairs_50'    :    load_pronoun_pairs_50_train ,
    'new_top_uppers'      :    load_new_top_uppers_train ,
    'top_7K_pair_freq'    :    load_top_7K_pair_freq_train ,
    'top_7K_x_None_freq'  :    load_top_7K_x_None_freq_train ,
    'pair_freq'           :    load_pair_freq_train ,
    'lstm'                :    load_lstm_train,
    'top_25_uppers':load_top_25_uppers_train,
    'diff_idf':load_diff_idf_train
}

names, n_estimators, subsample, colsample, max_depth

nr generic_submit_stacking_template.py diff_idf 500 0.7 0.7 5 &
done submit

nr generic_submit_stacking_template.py diff_idf,magic 500 0.7 0.7 5 &
done submit

nr generic_submit_stacking_template.py common_words,lengths,diff_idf,magic 500 0.7 0.7 5 &
done submit


nr generic_stacking_template.py lstm,diff_idf,magic 500 0.7 0.7 5 &
done

nr generic_stacking_template.py new_top_uppers 500 0.7 0.7 5 &
done

nr generic_stacking_template.py new_top_uppers,magic 300 0.7 0.7 5 &
done


nr generic_stacking_template.py new_top_uppers,magic 300 0.6 0.7 5 &
done


nr generic_stacking_template.py top_7K_pair_freq,magic,top_7K_x_None_freq 300 0.6 0.7 3 &
done

nr generic_stacking_template.py top_7K_pair_freq,top_7K_x_None_freq,pair_freq,pronoun_pairs_50 500 0.6 0.7 5 &
done

nr generic_stacking_template.py top_7K_pair_freq,top_7K_x_None_freq,pair_freq,pronoun_pairs_50,glove_metrics 400 0.6 0.7 4 &
done

nr generic_stacking_template.py lstm,magic,max_k_cores 400 0.6 0.7 3 &
done

nr generic_stacking_template.py diff_idf,magic,max_k_cores,top_7K_x_None_freq 500 0.6 0.7 3 &
done

nr generic_stacking_template.py new_top_uppers,pair_freq,lex_metrics 500 0.6 0.7 5 &
done

nr generic_stacking_template.py diff_idf,new_top_uppers,pair_freq,lex_metrics,lengths,common_words 400 0.6 0.7 4 &
done

nr generic_stacking_template.py top_7K_pair_freq,top_7K_x_None_freq 500 0.6 0.7 3 &
done

nr generic_stacking_template.py pronoun_pairs_50,aux_pairs_50,magic 300 0.6 0.7 4 &
done

nr generic_stacking_template.py top_7K_pair_freq,top_7K_x_None_freq,lstm 500 0.6 0.7 5 &
done

nr generic_stacking_template.py glove_metrics 500 0.6 0.7 5 &
done

nr generic_stacking_template.py glove_metrics,lstm,magic 500 0.6 0.7 5 &
done

nr generic_stacking_template.py pronoun_pairs_50,aux_pairs_50,magic,top_7K_pair_freq,top_7K_x_None_freq 300 0.6 0.7 4 &
done