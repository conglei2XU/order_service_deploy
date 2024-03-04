import multiprocessing
import os
import random
import time
import json
import pickle
from collections import defaultdict
from multiprocessing import Pool

import flask
import torch
import pandas as pd
from transformers import HfArgumentParser

from utilis.inference_wrapper import InferenceWrapper
from pipeline.arguments import CustomTrainArguments, ModelArguments, ControlArguments

TASKS = ['duty_reason', 'duty_depart', 'duty_dep_name', 'proc_type', 'keyword', 'location_info', 'manage_range']
BEST_CHECKPOINT = ['7000', '2000', '1000', '2000', '1000', '1000', '1000']
# TASKS = ['duty_reason', 'location_info']
# BEST_CHECKPOINT = ['7000', '1000']
SPECIAL_TASKS = ['duty_major']
CONFIG = 'config/config_Qinghai'
BEST_MODEL = 'saved_model/'


# pool = Pool(processes=len(TASKS))


def init_args():
    parser = HfArgumentParser((ControlArguments, ModelArguments, CustomTrainArguments))
    # control_arguments_, model_arguments_, train_arguments_ = ControlArguments, ModelArguments, CustomTrainArguments
    control_arguments_, model_arguments_, train_arguments_ = parser.parse_json_file(
        os.path.join(CONFIG, 'parameters', 'duty_reason.json'))
    return train_arguments_, model_arguments_, control_arguments_


def run_inference(task_idx):
    task_ = TASKS[task_idx]
    parameters_path = os.path.join(CONFIG, 'parameters')
    train_arguments, model_arguments, control_arguments = init_args()
    parameter_ = json.load(open(os.path.join(parameters_path, f'{task_}.json'), 'r'))
    best_model_path = os.path.join(BEST_MODEL, parameter_['task_name'])

    parameter_['code_table'] = os.path.join(CONFIG, task_, 'code_table.json')
    parameter_['best_model_path'] = os.path.join(best_model_path, 'pytorch_model.bin')
    parameter_['cached_tokenizer'] = f'cache/tokenizer.bin'
    parameter_['label_mapping'] = os.path.join(CONFIG, task_, 'label_mapping_cls.json')
    parameter_['do_inference'] = True

    run_model = InferenceWrapper(train_arguments, model_arguments, control_arguments, parameter_)
    # pred_word, pred_id = run_model.predict(serv_content, serv_type)
    return run_model


reason_to_major = json.load(open('config/reason_to_major.json', 'r'))

task_to_model = []
# load task specific parameters
for idx, task in enumerate(TASKS):
    task_to_model.append(run_inference(idx))


def run(serv_content, serv_type):
    # load task specific parameters
    result_list = []
    try:
        for i in range(len(TASKS)):
            result_list.append(task_to_model[i].predict(serv_content, serv_type))
    except Exception as e:
        print(f'An errors occurred: {e}')

    duty_reason_word, duty_reason_id = result_list[0]
    duty_depart_word, duty_depart_id = result_list[1]
    duty_dep_name_word, duty_dep_name_id = result_list[2]
    proc_type_word, proc_type_id = result_list[3]
    keyword_word, keyword_id = result_list[4]
    location_info_word, location_info_id = result_list[5]
    manage_range_word, manage_range_id = result_list[6]
    try:
        duty_major_id = reason_to_major[duty_reason_word[0]]
    except KeyError:
        duty_major_id = random.choice(list(reason_to_major.values()))
        print(f'Missing {duty_reason_id}')
    return duty_reason_id[0], keyword_id[0], str(int(manage_range_id[0])), duty_dep_name_id[0], str(int(duty_major_id)), str(int(duty_depart_id[0])), str(int(location_info_id[0])), proc_type_id[0]


def test():
    train_path = 'Dataset/order_Qinghai/train.csv'
    read_columns = ['serv_content', 'serv_type'] + TASKS
    test_data = pd.read_csv(train_path, usecols=read_columns)
    test_data = test_data.dropna(axis=0, subset=['serv_content', 'serv_type'])
    # test_data = test_data[:1000]
    correct = defaultdict(int)
    start = time.time()
    for idx_, row in test_data.iterrows():
        serv_content, serv_type = row['serv_content'], row['serv_type']
        # duty_reason_gold, duty_depart_gold, duty_dep_name_gold = row[TASKS[0]], row[TASKS[1]], row[TASKS[2]]
        # proc_type_gold, keyword_gold, lo
        predict_results = run(serv_content, serv_type)
        for id_, task_ in enumerate(TASKS):
            if predict_results[id_][0] == row[task_]:
                correct[task_] += 1
    print(time.time() - start)
    json.dump(correct, open('pred_result.json', 'w'))


if __name__ == "__main__":
    # pool = Pool(len(TASKS))
    test()
    # content = "此用户来电称办理此套餐时有1000元通话分钟数，现查到500分钟用户不认可，要求核实处理，谢谢！"
    # sheet = "投诉工单（2021版）>>移网>>【离网】离网（离网+拆机+销户）>>销户拆机状态播报与实际不符"
    # run(content, sheet)
