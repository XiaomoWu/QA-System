import spacy

nlp = spacy.load('en_core_web_lg')
doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion. What day is it today?')

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)



noun_chunks = ' '.join([n.text for n in doc.noun_chunks])
noun_chunks

for n in noun_chunks:
    print(n, type(n))
