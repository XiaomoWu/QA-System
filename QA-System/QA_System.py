#set working directory
#please change dir_path to where your solution located."""
dir_path = 'C:/Users/rossz/OneDrive/Academy/the U/Assignment/AssignmentSln/NLP-04-QA/QA-System/QA-System/'
os.chdir(dir_path)

import os
import re
import spacy
import pandas as pd
import qa_io
import qa_algo
from copy import deepcopy


class QA:
    def __init__(self):
        input_fpath = 'developset/input.txt'
        if not os.path.exists(input_fpath):
            qa_io.create_input(dir_path)

        self.input_dir, self.story_ids = qa_io.get_story_id_from_input(input_fpath)

        # create question and story dataset. Read from disk if they exist else create from scratch.
        self.story_data = (qa_io.get_story_data(self.story_ids, self.input_dir) if not os.path.exists('story_data.pkl') else pd.read_pickle('story_data.pkl'))
        self.question_and_ans_data = (qa_io.get_question_and_ans_data(self.story_ids, self.input_dir) if not os.path.exists('question_and_ans_data.pkl') else pd.read_pickle('question_and_ans_data.pkl'))

    # produce answer
    def _extract_answer(self):
        self.question_and_ans_data = qa_algo.extract_answer(self.story_data, self.question_and_ans_data, self.story_ids)


    # score output
    def _score(self):
        self.question_and_ans_data = qa_io.score(self.question_and_ans_data)

qa = QA()
qa._extract_answer()
qa._score()
#ans = qa.question_and_ans_data
#ans.to_csv('ans.csv', index = False)

        


