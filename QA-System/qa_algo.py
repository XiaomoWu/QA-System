import os
import re
import spacy
import pandas as pd
from copy import deepcopy

nlp = spacy.load('en_core_web_lg')

def extract_answer(story_data, question_and_ans_data, story_ids):
    """ (1) get answer, then modify self.question_and_ans_data by add the answer to it. 
        (2) for each story id, extract its question, then look up in story_data, find the best sentence"""

    for story_id in story_ids:
        story = story_data.loc[lambda df: df.story_id == story_id, 'story'].values[0]
        question_ids = question_and_ans_data.loc[lambda df: df.story_id == story_id, 'question_id']

        for question_id in question_ids:
            # get the question and answer
            question = question_and_ans_data.loc[lambda df: df.question_id == question_id, 'question'].values[0]
            if 'answer' in question_and_ans_data:
                answer = question_and_ans_data.loc[lambda df: df.question_id == question_id, 'answer'].values[0]

            ans = []
            for sent in story.sents:
                sim = sent.similarity(question)
                ans.append({'question_id': question_id, 'answer_pred': sent, 'similarity': sim})

            ans = pd.DataFrame(ans).reindex(['question_id', 'answer_pred', 'similarity'], axis = 1)
            ans.sort_values(by = ['similarity'], ascending = False, inplace = True)

            question_and_ans_data.loc[lambda df: df.question_id == question_id, 'answer_pred'] = ans.iloc[0]['answer_pred'].text

    question_and_ans_data['answer_pred'] = question_and_ans_data['answer_pred'].apply(nlp)

    return question_and_ans_data
