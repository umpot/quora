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

#nr stacking_lstm.py 0 lemmas glove yes &
#nr stacking_lstm.py 1 lemmas glove yes &
#nr stacking_lstm.py 2 lemmas glove yes &
#nr stacking_lstm.py 3 lemmas glove yes &
#nr stacking_lstm.py 4 lemmas glove yes &
#
#nr submit_stacking_lstm.py lemmas glove yes &


#nr stacking_lstm.py 0 nouns glove no &
#nr stacking_lstm.py 1 nouns glove no &
#nr stacking_lstm.py 2 nouns glove no &
#nr stacking_lstm.py 3 nouns glove no &
#nr stacking_lstm.py 4 nouns glove no &
#
#nr submit_stacking_lstm.py nouns glove no &


nr stacking_lstm.py 0 verbs glove no &
nr stacking_lstm.py 1 verbs glove no &
nr stacking_lstm.py 2 verbs glove no &
nr stacking_lstm.py 3 verbs glove no &
nr stacking_lstm.py 4 verbs glove no &

nr submit_stacking_lstm.py verbs glove no &