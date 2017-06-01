import subprocess
scripts = [
    # 'submit_stacking_no_emb_simpl_idf_light.py',
    # 'submit_stacking_no_metrics_light.py',
    # 'submit_stacking_no_tfidf_light.py',
    # 'submit_stacking_no_top_tokens_light.py',
    # 'submit_stacking_only_glove_emb_light.py',
    'submit_stacking_only_word2vec_emb_light.py',
    'submit_stacking_only_lex_emb_light.py'

]


for sc_name in scripts:
    subprocess.call(['nohup', 'python', '-u', sc_name])
