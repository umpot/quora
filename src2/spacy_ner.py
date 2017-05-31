import spacy
import re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

nlp = spacy.load('en', parser=False)

txt = u'How did Darth Vader fought Darth Maul in Star Wars Legends?'
doc = nlp(txt)

for word in doc:
    print(word.text, word.lemma, word.lemma_, word.tag, word.tag_, word.pos, word.pos_)

for e in doc.ents:
    print e.start, e.end, e.label_, str(e)


counter=0

def process_df(df, cols):
    def postag_and_ner(s):
        global counter
        counter+=1
        if counter%1000 == 0:
            print counter
        doc=nlp(str(s).decode("utf-8"))
        pos = [[word.text, word.lemma, word.lemma_, word.tag, word.tag_, word.pos, word.pos_] for word in doc]
        ner = [[e.start, e.end, e.label_, str(e)] for e in doc.ents]
        return [pos, ner]
    for col in cols:
        df['nlp_{}'.format(col)]=df[col].apply(postag_and_ner)
    df.to_json('nlp_processed_test.json')


def normalize_str(s):
    return ' '.join(re.sub("[^a-zA-Z0-9]", " ", s).split()).lower()


def store_lemmas_df(df, fp, index_label):
    df['lemmas_q1'] = df['nlp_question1'].apply(lambda s: ' '.join([x[2] for x in s[0]]))
    df['lemmas_q1'] = df['lemmas_q1'].apply(normalize_str)

    df['lemmas_q2'] = df['nlp_question2'].apply(lambda s: ' '.join([x[2] for x in s[0]]))
    df['lemmas_q2'] = df['lemmas_q2'].apply(normalize_str)
    df[['lemmas_q1','lemmas_q2']].to_csv(fp, index_label=index_label)


def store_tokens_df(df, fp, index_label):
    df['tokens_q1'] = df['nlp_question1'].apply(lambda s: ' '.join([x[0] for x in s[0]]))
    df['tokens_q1'] = df['tokens_q1'].apply(normalize_str)

    df['tokens_q2'] = df['nlp_question2'].apply(lambda s: ' '.join([x[0] for x in s[0]]))
    df['tokens_q2'] = df['tokens_q2'].apply(normalize_str)

    df[['tokens_q1','tokens_q2']].to_csv(fp, index_label=index_label)
