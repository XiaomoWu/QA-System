doc = nlp(u"what's the time now Nov 12 today ? who are you? when did you go there?")
[ent.text for ent in doc.ents if ent.label_ in ['DATE', 'TIME']]