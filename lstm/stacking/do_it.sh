#!/usr/bin/env bash


#nr stacking_lstm.py 0 lemmas word2vec yes &
#nr stacking_lstm.py 1 lemmas word2vec yes &
#nr stacking_lstm.py 2 lemmas word2vec yes &
#nr stacking_lstm.py 3 lemmas word2vec yes &
#nr stacking_lstm.py 4 lemmas word2vec yes &
#
#nr submit_stacking_lstm.py lemmas word2vec yes &


#nr stacking_lstm.py 0 question lex yes &
#nr stacking_lstm.py 1 question lex yes &
#nr stacking_lstm.py 2 question lex yes &
#nr stacking_lstm.py 3 question lex yes &
#nr stacking_lstm.py 4 question lex yes &
#
#nr submit_stacking_lstm.py question lex yes &

nr stacking_lstm.py 0 lemmas glove yes &
nr stacking_lstm.py 1 lemmas glove yes &
nr stacking_lstm.py 2 lemmas glove yes &
nr stacking_lstm.py 3 lemmas glove yes &
nr stacking_lstm.py 4 lemmas glove yes &

nr submit_stacking_lstm.py lemmas glove yes &