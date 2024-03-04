import os
import json
from json import JSONDecodeError

from flask import Flask, request, jsonify
import pandas as pd
from transformers import HfArgumentParser

from utilis.inference_wrapper import InferenceWrapper
from pipeline.arguments import CustomTrainArguments, ModelArguments, ControlArguments

app = Flask(__name__)

TASKS = ['duty_depart', 'manage_range']
BEST_CHECKPOINT = ['2000', '1000']
CONFIG = 'config/config_Qinghai'
BEST_MODEL = 'saved_model/'


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


task_to_model = []
# load task specific parameters
for idx_, task_ in enumerate(TASKS):
    task_to_model.append(run_inference(idx_))


@app.route('/service', methods=['POST'])
def service():
    response_info = {'STATUS': '0000', 'MSG': 'SUCCESS', 'RSP': {'RSP_CODE': '0000',
                                                                 'RSP_DESC': '业务请求正常', 'DATA': []}}
    # do initialization
    duty_depart_id = ''
    mange_range_id = ''
    duty_depart_word = ''
    manage_range_word = ''

    try:
        data = json.loads(request.get_data())
        serv_content = data['serv_content']
        serv_type = data['serv_type']
        result_list = []
        for idx, task in enumerate(TASKS):
            result_list.append(task_to_model[idx].predict(serv_content, serv_type))
        duty_depart_word, duty_depart_id = result_list[0]  # [移网质量>>网络覆盖>>弱覆盖需优化解决>>弱覆盖需优化解决], [943fb1cb08bb4d3faf3047ab088e5b17_S2]
        manage_range_word, mange_range_id = result_list[1]

    except JSONDecodeError as e:
        response_info['RSP']['RSP_CODE'] = '0001'
        response_info['RSP']['RSP_DESC'] = f'业务参数错误2,具体如下：{repr(e)}'
    except KeyError as e:
        response_info['RSP']['RSP_CODE'] = '0001'
        response_info['RSP']['RSP_DESC'] = f'业务参数错误3,具体如下：{repr(e)}'
    except Exception as e:
        response_info['RSP']['RSP_CODE'] = '0002'
        response_info['RSP']['RSP_DESC'] = f'业务请求异常4,具体如下：{repr(e)}'
    return_data = {
        'duty_depart_id': duty_depart_id,
        'manage_range_id': mange_range_id,
        'duty_depart_word': duty_depart_word,
        'manage_range_word': manage_range_word
    }
    response_info['RSP']['DATA'] = [return_data]
    return jsonify(response_info), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10081)
