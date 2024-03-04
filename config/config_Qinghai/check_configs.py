import os
import json

def check_():
    list_file = os.listdir()
    print(list_file)
    for file in list_file:
        if '_' in file and file != 'serv_type':
            if os.path.isdir(file):
                labels = os.path.join(file, 'label_mapping_cls.json')
                codes = os.path.join(file, 'code_table.json')
                labels_dict = json.load(open(labels, 'r'))
                codes_dict = json.load(open(codes, 'r'))
                if len(labels_dict) != len(codes_dict):
                    print(file)


if __name__ == "__main__":
    check_()