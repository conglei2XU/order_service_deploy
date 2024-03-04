import json
import time
import datetime
from collections import defaultdict
import requests

import pandas as pd

TASKS = ['duty_reason', 'duty_depart', 'duty_dep_name', 'proc_type', 'keyword', 'location_info', 'manage_range']
BEST_CHECKPOINT = ['7000', '6000', '4000', '1000', '1000', '1000', '2000']

POST_ = {"proId": "81",
         "complArea": "", "busType": "", "answerContent": "", "actionView": "", "asrText": ""}
URL = 'http://localhost:5000/duty'

KEY_MAP = {'codeComplainCategory': 'duty_reason_id',
           'keyword': 'keyword_id',
           'manageRange': 'manage_range_id',
           'dutyDepName': 'duty_dep_name_id',
           'codeDutyMajor': 'duty_major_id',
           'dutyDepart': 'duty_depart_id',
           'locationInfo': 'location_info_id',
           'procType': 'proc_type_id'}


# if 'codeComplainCategory' in KEY_MAP:
#     print(True)

def test():
    train_path = 'Dataset/order_Qinghai/train.csv'
    read_columns = ['serv_content', 'serv_type'] + TASKS
    test_data = pd.read_csv(train_path)
    test_data = test_data.dropna(axis=0, subset=['serv_content', 'serv_type', 'location_info'])
    test_data = test_data[:100]
    correct = defaultdict(int)
    start = time.time()
    for idx_, row in test_data.iterrows():
        serv_content, serv_type = row['serv_content'], row['serv_type']
        POST_['sheetTypePath'] = serv_type
        POST_['servContent'] = serv_content
        # duty_reason_gold, duty_depart_gold, duty_dep_name_gold = row[TASKS[0]], row[TASKS[1]], row[TASKS[2]]
        # proc_type_gold, keyword_gold, lo
        response = requests.post(URL, data=json.dumps(POST_))
        if int(response.status_code) == 200:
            response_ = json.loads(response.text)
            resp_code = response_['RSP']['RSP_CODE']
            pred_result = response_['RSP']['DATA'][0]
            if resp_code == '0000':
                for key, pred_value in pred_result.items():
                    try:
                        corr_key = KEY_MAP[key]
                        gold_value = row[corr_key]
                    except KeyError:
                        # print(f'Missing key {corr_key} {key}')
                        print(key, pred_value)
                        exit(1)
                    if not isinstance(gold_value, str):
                        try:
                            gold_value = str(int(gold_value))
                        except ValueError:
                            gold_value = str(gold_value)
                    if gold_value == pred_value:
                        correct[corr_key] += 1
            else:
                print(f'Do prediction errors: {serv_content}, {idx_}')

        else:
            print(response.status_code)
            exit(1)

    duration = time.time() - start
    format_duration = datetime.timedelta(seconds=duration)
    print(format_duration)
    json.dump(correct, open('pred_result.json', 'w'))


if __name__ == "__main__":
    test()
