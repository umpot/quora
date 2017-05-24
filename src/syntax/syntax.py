from nltk import Tree
import spacy
import textacy
nlp = spacy.load('en')

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
       return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

def print_tree(s):
    doc = nlp(s)
    print [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]

def extract_subj_verb(s):
    doc=nlp(s)
    return [x for x in textacy.extract.subject_verb_object_triples(doc)]