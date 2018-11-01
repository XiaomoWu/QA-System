import os
import re
import spacy
import pandas as pd
from copy import deepcopy
nlp = spacy.load('en_core_web_lg')

def get_story_id_from_input(input_fpath):
    with open(input_fpath) as f:
        lines = f.readlines()
    input_dir = lines[0].strip()
    story_id = [l.strip() for l in lines[1:]]
    return (input_dir, story_id)

def get_story_data(story_ids, input_dir):
    stories = []
    for s in story_ids:
        with open('%s%s.story' % (input_dir[1:], s)) as f:
        #with open('developset/1999-W02-5.story') as f:
            story = f.read()
            m = re.search(r'HEADLINE:(.+)\n', story)
            if m: 
                headline = m.group(1).strip()
            else: 
                print('NO HEADLINE!')
            m = re.search(r'DATE:(.+)\n', story)
            if m: 
                date = m.group(1).strip()
            else:
                print('NO DATE!')
            m = re.search(r'TEXT:([\s\S]+)', story)
            if m:
                storytxt = nlp(m.group(1).strip())
            else:
                print('NO STORY CONTENT!')

            stories.append({'story_id': s, 'headline': headline, 'date': date, 'story': storytxt})
                
    df = pd.DataFrame(stories).reindex(['story_id', 'headline', 'date', 'story'], axis = 1)
        
    # write to disk
    #df.to_pickle('story_data.pkl')

    return df

def get_question_and_ans_data(story_ids, input_dir, has_ans = True):
    """ if has_ans = True, read ".answers" files; intead, read ".questions" files
    """
    questions = []
    for s in story_ids:
        if has_ans == False:
            p = '%s%s.questions' % (input_dir[1:], s)
        elif has_ans == True:
            p = '%s%s.answers' % (input_dir[1:], s)
        with open(p) as f:
            lines = f.read()
        for q in lines.split('\n\n'):
            if q.strip() != '':
                m = re.search(r'QuestionID: *(\S+)\n?', q)
                if m:
                    question_id = m.group(1).strip()
                else:
                    print('NO QUESTION ID!')
                m = re.search(r'Question:(.+)\n?', q)
                if m:
                    question = nlp(m.group(1).strip())
                else:
                    print('NO QUESTION!')
                m = re.search(r'Difficulty:(.+)\n?', q)
                if m:
                    difficulty = m.group(1).strip()
                else:
                    print('NO DIFFICULTY!')
                m = re.search(r'Answer:(.+)\n?', q)
                if m:
                    answer = nlp(m.group(1).strip())
                else:
                    print('NO ANSWER!')

                questions.append({'story_id': s, 'question_id': question_id, 'question': question, 'difficulty': difficulty, 'answer': answer})

    df = pd.DataFrame(questions).reindex(['story_id', 'question_id', 'question', 'difficulty', 'answer'], axis = 1)

    # write to disk
    #df.to_pickle('question_and_ans_data.pkl')
    return df


def create_input():
    """ create an input file (list file names of stories)
        required by pp. 3 of the instruction"""

    fpath = os.listdir(dir_path + 'developset')

    # list the file path of all answers and questions
    story_fpath = [f for f in fpath if os.path.splitext(f)[1] == '.story']
    story_fpath.sort()
    answer_fpath = [f for f in fpath if os.path.splitext(f)[1] == '.answers']
    answer_fpath.sort()
    question_fpath = [f for f in fpath if os.path.splitext(f)[1] == '.questions']
    question_fpath.sort()

    with open('developset/input.txt', 'w') as f:
        f.writelines('/developset/\n')
        for s in story_fpath:
            f.writelines(os.path.splitext(s)[0])
            f.writelines('\n')


