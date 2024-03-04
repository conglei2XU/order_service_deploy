import json
from json import JSONDecodeError
from flask import Flask, request, jsonify



app = Flask(__name__)

from order_service_backend2 import run

SUPPORT_PROVINCE = ['81']


def after_request(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Credentials'] = 'true'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'x-Requested-With, Content-Type, cbss-ms-gray'
    resp.headers['Access-Control-Max-Age'] = '3600'
    return resp


app.after_request(after_request)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = 'application/json;charset=utf-8'  # 指定浏览器渲染的文件类型，和解码格式；


@app.route('/')
def hello_world():
    return ('欢迎智能中心AI平台！智能中心AI平台，以NLP和CV原子能力为底座，结合具体的业务需求;'
            '通过有机地组织和整合原子能力构建模型，实现业务场景快速高质量支撑。目前AI平台上线原子能力30+，'
            '覆盖工单分类、意图识别、文本摘要等业务场景，利用AI技术持续赋能业务，助力公司业务数字化转型')


@app.route('/duty', methods=['POST'])
def duty():
    response_info = {'STATUS': '0000', 'MSG': 'SUCCESS', 'RSP': {'RSP_CODE': '0000',
                                                                 'RSP_DESC': '业务请求正常', 'DATA': []}}
    # do initialization
    codeComplainCategory_id = ''
    keyword_id = ''
    manageRange_id = ''
    dutyDepName_id = ''
    codeDutyMajor_id = ''
    dutyDepart_id = ''
    locationInfo_id = ''
    procType_id = ''
    msg = ''
    return_data_empty = {
        'codeComplainCategory': codeComplainCategory_id,
        'keyword': keyword_id,
        'manageRange': manageRange_id,
        'dutyDepName': dutyDepName_id,
        'codeDutyMajor': codeDutyMajor_id,
        'dutyDepart': dutyDepart_id,
        'locationInfo': locationInfo_id,
        'procType': procType_id
    }
    try:
        data = json.loads(request.get_data())
        pro_id = data['proId']
        sheet_type_path = data['sheetTypePath']
        serv_content = data['servContent']
        complArea = data['complArea']
        busType = data['busType']
        answerContent = ''
        action_view = ''
        asr_text = ''
        if 'answerContent' in data:
            answerContent = data['answerContent']
        if 'actionView' in data:
            action_view = data['actionView']
        if 'asrText' in data:
            asr_text = data['asrText']
        auxiliary_info = '#'.join([sheet_type_path, answerContent, action_view, asr_text])
        if len(pro_id) == 0:
            msg = 'pro_id字段为空，请检查输入内容'
            response_info['RSP']['RSP_CODE'] = '0001'
            response_info['RSP']['RSP_DESC'] = f'业务参数错误1,具体如下：{msg}'
            # return codeComplainCategory_id, keyword_id, manageRange_id, dutyDepName_id, codeDutyMajor_id, dutyDepart_id, locationInfo_id, procType_id, msg
        elif serv_content.strip() and sheet_type_path.strip():
            codeComplainCategory_id, keyword_id, manageRange_id, dutyDepName_id, codeDutyMajor_id, dutyDepart_id, locationInfo_id, procType_id = \
                run(serv_content, sheet_type_path)
            if codeComplainCategory_id and keyword_id and manageRange_id and dutyDepName_id and codeDutyMajor_id and dutyDepart_id and locationInfo_id and procType_id:

                return_data = {
                    'codeComplainCategory': codeComplainCategory_id,
                    'keyword': keyword_id,
                    'manageRange': manageRange_id,
                    'dutyDepName': dutyDepName_id,
                    'codeDutyMajor': codeDutyMajor_id,
                    'dutyDepart': dutyDepart_id,
                    'locationInfo': locationInfo_id,
                    'procType': procType_id
                }
                # print(codeComplainCategory_id, keyword_id, manageRange_id, procType_id, dutyDepName_id, codeDutyMajor_id, msg)
                response_info['RSP']['DATA'] = [return_data]
            else:
                response_info['RSP']['DATA'] = [return_data_empty]
                response_info['RSP']['RSP_CODE'] = '0001'
                response_info['RSP']['RSP_DESC'] = f'模型预测错误'
        else:
            msg = 'serv_type or sheet_path 为空'
            response_info['RSP']['RSP_CODE'] = '0001'
            response_info['RSP']['RSP_DESC'] = f'业务参数错误5,具体如下：{msg}'
        # if codeComplainCategory_id and keyword_id and manageRange_id and procType_id and dutyDepName_id and codeDutyMajor_id:

    except JSONDecodeError as e:
        response_info['RSP']['RSP_CODE'] = '0001'
        response_info['RSP']['RSP_DESC'] = f'业务参数错误2,具体如下：{repr(e)}'
    except KeyError as e:
        response_info['RSP']['RSP_CODE'] = '0001'
        response_info['RSP']['RSP_DESC'] = f'业务参数错误3,具体如下：{repr(e)}'
    except Exception as e:
        response_info['RSP']['RSP_CODE'] = '0002'
        response_info['RSP']['RSP_DESC'] = f'业务请求异常4,具体如下：{repr(e)}'
    if response_info['RSP']['RSP_CODE'] != '0000':
        response_info['RSP']['DATA'] = [return_data_empty]
    return jsonify(response_info), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10081)
