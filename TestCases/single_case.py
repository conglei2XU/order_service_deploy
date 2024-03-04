# copyright 2021 The Chinaunicom Software Team. All rights reserved.
# @FileName: test_url.py
# @Author  : Dammy
# @Time    : 2021/8/30

# coding:utf-8
import random
import re, os, json, time
import requests

url = 'http://localhost:5000/duty'


def api_work():
    test_data = {"proId": "81", "sheetTypePath": "未知", "servContent":
        "融合,离网,离网,离网,拆机,销户,无法,销户,融合,业务,成都,用户,来电,要求,将,此卡,绑定,宽带,销户,详见,按照,工单,结果,解释,不认可 ",
                 "complArea": "", "busType": "", "answerContent": "", "actionView": "", "asrText": ""}
    # url = 'http://10.188.103.42:8687/gd-intent-classifer/intent_server'
    # url = 'http://10.188.48.148:8687/gd-intent-classifer/intent_server'
    # url = 'http://10.233.87.28:9991/intent_server'
    begin = time.time()
    r = requests.post(url, data=json.dumps(test_data))
    result_data = json.loads(r.text)
    if result_data['RSP']['RSP_CODE'] != '0000':
        print(result_data)
    assert result_data['RSP']['RSP_CODE'] == '0000'
    assert len(result_data['RSP']['DATA'][0]) == 8
    for key, value in result_data['RSP']['DATA'][0].items():
        assert value
    print(result_data)


def invalid_input():
    print('Invalid_input1: ')
    test_data = {"proId": "83", "sheetTypePath": "", "servContent":
        "融合,离网,离网,离网,拆机,销户,无法,销户,融合,业务,成都,用户,来电,要求,将,此卡,绑定,宽带,销户,详见,按照,工单,结果,解释,不认可 ",
                 "complArea": "", "busType": "", "answerContent": "", "actionView": "", "asrText": ""}
    r = requests.post(url, data=json.dumps(test_data))
    result_data = json.loads(r.text)
    if result_data['RSP']['RSP_CODE'] != '0000':
        print(result_data)
    assert result_data['RSP']['RSP_CODE'] == '0001'
    assert len(result_data['RSP']['DATA'][0]) == 8
    print('Invalid_input2: ')
    test_data = {"proId": "81", "sheetTypePath": "", "servContent":
        "",
                 "complArea": "", "busType": "", "answerContent": "", "actionView": "", "asrText": ""}
    r = requests.post(url, data=json.dumps(test_data))
    result_data = json.loads(r.text)
    assert result_data['RSP']['RSP_CODE'] == '0001'
    assert len(result_data['RSP']['DATA'][0]) == 8
    print(result_data)
    test_data = {"proId": "81", "servContent":
        "",
                 "complArea": "", "busType": "", "answerContent": "", "actionView": "", "asrText": ""}
    r = requests.post(url, data=json.dumps(test_data))
    result_data = json.loads(r.text)
    assert result_data['RSP']['RSP_CODE'] == '0001'
    assert len(result_data['RSP']['DATA'][0]) == 8
    print(result_data)


if __name__ == "__main__":
    invalid_input()
    api_work()

