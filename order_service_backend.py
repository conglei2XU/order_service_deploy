import os
import random
import time
import json
import concurrent.futures
from collections import defaultdict
import requests

import flask
import torch
import pandas as pd
from transformers import HfArgumentParser

from utilis.inference_wrapper import InferenceWrapper
from pipeline.arguments import CustomTrainArguments, ModelArguments, ControlArguments

TASKS = ['duty_reason', 'location_info', 'duty_dep_name', 'proc_type', 'keyword', 'duty_depart', 'manage_range']
BEST_CHECKPOINT = ['7000', '1000', '4000', '1000', '1000', '6000', '2000']
# TASKS = ['duty_reason', 'location_info']
# BEST_CHECKPOINT = ['7000', '1000']
SPECIAL_TASKS = ['duty_major']
CONFIG = 'config/config_Qinghai'
BEST_MODEL = 'saved_model/Qinghai'

services = ['10082', '10083', '10084']

pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(services))
reason_to_major = json.load(open('config/reason_to_major.json', 'r'))

URL = f'127.0.0.1:/service'


# load task specific parameters

def do_request(url, data):
    response = requests.post(url, data=data)
    response_data = json.loads(response.text)
    response_ans = response_data['RSP']['DATA'][0]
    return response_ans


def run(serv_content, serv_type):
    # load task specific parameters
    result_list = []
    request_data = {'serv_content': serv_content, 'serv_type': serv_type}
    request_data_raw = json.dumps(request_data)
    try:
        for i, service_port in enumerate(services):
            url = f'127.0.0.1:{service_port}/service'
            result_list.append(pool.submit(do_request, url, request_data_raw))
            # result_list.append(pool.submit(paw_, 2, 1))

    except Exception as e:
        print(f'An errors occurred: {e}')
    finally:
        pass
    duty_reason_word, duty_reason_id = result_list[0].get()['duty_reason_word'], result_list[0].get()['duty_reason_id']
    location_info_word, location_info_id = result_list[0].get()['location_info_word'], result_list[0].get()['location_info_id']
    duty_dep_name_word, duty_dep_name_id = result_list[1]['duty_dep_name_word'].get(), result_list[1].get()['duty_dep_name_id']
    proc_type_word, proc_type_id = result_list[1].get()['proc_type_word'], result_list[1].get()['proc_type_id']
    keyword_word, keyword_id = result_list[1].get()['keyword_word'], result_list[1].get()['keyword_id']
    duty_depart_word, duty_depart_id = result_list[2].get()['duty_depart_word'], result_list[2].get()['duty_depart_id']
    manage_range_word, manage_range_id = result_list[2].get()['manage_range_word'], result_list[2].get()['manage_range_id']
    try:
        duty_major_id = reason_to_major[duty_reason_word[0]]
    except KeyError:
        duty_major_id = random.choice(list(reason_to_major.values()))
        print(f'Missing {duty_reason_id}')

    return duty_reason_id[0], keyword_id[0], str(int(manage_range_id[0])), duty_dep_name_id[0], str(
        int(duty_major_id)), str(int(duty_depart_id[0])), str(int(location_info_id[0])), proc_type_id[0]


def main():
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
        for i in predict_results:
            print(i.result())
        # for id_, task_ in enumerate(TASKS):
            # print(predict_results[])
            # pass
            # if predict_results[id_][0] == row[task_]:
            #     correct[task_] += 1
    print(time.time() - start)
    json.dump(correct, open('pred_result.json', 'w'))


if __name__ == "__main__":
    # pool = Pool(len(TASKS))
    main()
    # content = "此用户来电称办理此套餐时有1000元通话分钟数，现查到500分钟用户不认可，要求核实处理，谢谢！"
    # sheet = "投诉工单（2021版）>>移网>>【离网】离网（离网+拆机+销户）>>销户拆机状态播报与实际不符"
    # run(content, sheet)
