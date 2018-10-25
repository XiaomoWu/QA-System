import os

# set working directory
dir_path = 'C:/Users/Yu Zhu/OneDrive/Academy/the U/Assignment/AssignmentSln/NLP-04-QA/QA-System/QA-System/'
os.chdir(dir_path)

'''
create an input file (list file names of stories)
'''
def create_input():
    fpath = os.listdir(dir_path + 'developset')

    # list the file path of all answers and questions
    story_fpath = [f for f in fpath if os.path.splitext(f)[1] == '.story']
    story_fpath.sort()
    answer_fpath = [f for f in fpath if os.path.splitext(f)[1] == '.answers']
    answer_fpath.sort()
    question_fpath = [f for f in fpath if os.path.splitext(f)[1] == '.questions']
    question_fpath.sort()

    with open('input.txt', 'w') as f:
        f.writelines('/developset/')
        for s in story_fpath:
            f.writelines(os.path.splitext(s)[0])
            f.writelines('\n')

    # print questions
    for a in answer_fpath:
        with open(dir_path + 'developset/' + a) as f:
            line = f.readlines()
            print(line)

create_input()


