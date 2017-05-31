#!/usr/bin/env bash

#process_paralell(train_test, embed_name, operation, type_of_cols)
#['wmd', 'norm_wmd', 'metrics', 'combine']
#['tokens', 'lemmas']

#nr glove.py test glove wmd tokens       &
#nr glove.py test glove wmd lemmas       &
#
#nr glove.py test glove norm_wmd tokens  &
#nr glove.py test glove norm_wmd lemmas  &
#
#nr glove.py test glove metrics tokens   &
#nr glove.py test glove metrics lemmas   &
#
#nr glove.py test glove combine lemmas   &


#nr glove.py test word2vec wmd tokens       &
#nr glove.py test word2vec wmd lemmas       &
#
#nr glove.py test word2vec norm_wmd tokens  &
#nr glove.py test word2vec norm_wmd lemmas  &
#
#nr glove.py test word2vec metrics tokens   &
#nr glove.py test word2vec metrics lemmas   &
#
#nr glove.py test word2vec combine lemmas   &

#nr glove.py train word2vec wmd tokens       &
#nr glove.py train word2vec wmd lemmas       &
#
#nr glove.py train word2vec norm_wmd tokens  &
#nr glove.py train word2vec norm_wmd lemmas  &
#
#nr glove.py train word2vec metrics tokens   &
#nr glove.py train word2vec metrics lemmas   &
#
#nr glove.py train word2vec combine lemmas   &



#nr glove.py train lex wmd tokens       &
#nr glove.py train lex wmd lemmas       &
#
#nr glove.py train lex norm_wmd tokens  &
#nr glove.py train lex norm_wmd lemmas  &
#
#nr glove.py train lex metrics tokens   &
#nr glove.py train lex metrics lemmas   &
#
#nr glove.py train lex combine lemmas   &


nr glove.py test lex wmd tokens       &
nr glove.py test lex wmd lemmas       &

nr glove.py test lex norm_wmd tokens  &
nr glove.py test lex norm_wmd lemmas  &

nr glove.py test lex metrics tokens   &
nr glove.py test lex metrics lemmas   &

nr glove.py test lex combine lemmas   &