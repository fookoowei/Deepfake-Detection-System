import os

file_name = 'real_train.txt'  

current_directory = os.path.dirname(__file__)

file_path = os.path.join(current_directory, file_name)

with open(file_path, 'r') as fnr:
    lines = fnr.readlines()

modified_lines = ['PREFIX ' + line.strip() + ': [0]\n' for line in lines]

with open(file_path, 'w') as fnw:
    fnw.writelines(modified_lines)