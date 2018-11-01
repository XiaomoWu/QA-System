import os
import re
import spacy
import pandas as pd
from copy import deepcopy
from qa_io import *

#nlp = spacy.load('en_core_web_lg')

##set working directory
##please change dir_path to where your solution located."""
#dir_path = 'C:/Users/rossz/OneDrive/Academy/the U/Assignment/AssignmentSln/NLP-04-QA/QA-System/QA-System/'
#os.chdir(dir_path)

#create_input()

class QA:
    def __init__(self):
        self.input_fpath = 'developset/input.txt'
        self.input_dir, self.story_ids = get_story_id_from_input(self.input_fpath)

        # create question and story dataset. Read from disk if they exist else create from scratch.
        self.story_data = (get_story_data(self.story_ids, self.input_dir) if not os.path.exists('story_data.pkl') else pd.read_pickle('story_data.pkl'))
        self.question_and_ans_data = (get_question_and_ans_data(self.story_ids, self.input_dir) if not os.path.exists('question_and_ans_data.pkl') else pd.read_pickle('question_and_ans_data.pkl'))


    def _extract_answer(self):
        """ (1) get answer, then modify self.question_and_ans_data by add the answer to it. 
            (2) for each story id, extract its question, then look up in story_data, find the best sentence"""

        for story_id in self.story_ids:
            story = self.story_data.loc[lambda df: df.story_id == story_id, 'story'].values[0]
            question_ids = self.question_and_ans_data.loc[lambda df: df.story_id == story_id, 'question_id']

            for question_id in question_ids:
                # get the question and answer
                question = self.question_and_ans_data.loc[lambda df: df.question_id == question_id, 'question'].values[0]
                if 'answer' in self.question_and_ans_data:
                    answer = self.question_and_ans_data.loc[lambda df: df.question_id == question_id, 'answer'].values[0]

                ans = []
                for sent in story.sents:
                    sim = sent.similarity(question)
                    ans.append({'question_id': question_id, 'answer_pred': sent, 'similarity': sim})

                ans = pd.DataFrame(ans).reindex(['question_id', 'answer_pred', 'similarity'], axis = 1)
                ans.sort_values(by = ['similarity'], ascending = False, inplace = True)

                self.question_and_ans_data.loc[lambda df: df.question_id == question_id, 'answer_pred'] = ans.iloc[0]['answer_pred'].text

        self.question_and_ans_data['answer_pred'] = self.question_and_ans_data['answer_pred'].apply(nlp)

    def _score(self):
        def _make_precision(row):
            precision = len(_overlap_tokens(row['answer'], row['answer_pred'])) / len(row['answer_pred'])
            return precision

        def _make_recall(row):
            recall = len(_overlap_tokens(row['answer'], row['answer_pred'])) / len(row['answer'])
            return recall

        def _make_overlap(row):
            overlap = _overlap_tokens(row['answer'], row['answer_pred'])
            return overlap

        def _overlap_tokens(doc, other_doc):
            """Get the tokens from the original Doc that are also in the comparison Doc.
            """
            overlap = []
            other_tokens = [token.text for token in other_doc]
            for token in doc:
                if token.text in other_tokens:
                    overlap.append(token)
            return overlap
        
        self.question_and_ans_data['precision'] = self.question_and_ans_data.apply(_make_precision, axis = 1)
        self.question_and_ans_data['recall'] = self.question_and_ans_data.apply(_make_recall, axis = 1)
        self.question_and_ans_data['overlap'] = self.question_and_ans_data.apply(_make_overlap, axis = 1)
        self.question_and_ans_data = self.question_and_ans_data.assign(f_score = lambda x: (2 * x['precision'] * x['recall']) / (x['precision'] + x['recall']))

qa = QA()
qa._extract_answer()
qa._score()
ans = qa.question_and_ans_data
ans
ans.to_csv('ans.csv', index = False)

        


