import re

def normalize_str(s):
    try:
        return ' '.join(re.sub("[^a-zA-Z0-9]", " ", s).split()).lower()
    except:
        print 'blja'
        print type(s)
        print s
        print '==============='
        return ''


def normalize_and_store_df(df, fp):
    df = df[['question1', 'question2']]
    df['norm1'] = df['question1'].apply(normalize_str)
    df['norm2'] = df['question2'].apply(normalize_str)
    df[['norm1', 'norm2']].to_csv(fp, index_label='id')