SOS_token = 0
EOS_token = 1
MAX_LENGTH = 50
LABEL_PAD = -100
# LABEL_PAD = 2
THRESHOLD = 2
UNK = 1
PAD = 0
MAX_SENT_LENGTH = 300
END_TEXT_LEN = 200
# KEYS = ['serv_content', 'serv_type', 'duty_reason']
# KEYS = ['serv_content', 'serv_type', 'duty_dep_name']
# KEYS = ['serv_content', 'duty_dep_name']
# KEYS = ['serv_content', 'duty_reason']
KEYS = ['serv_content', 'location_info']
# KEYS = ['serv_content', 'serv_type', 'location_info']
# KEYS = ['受理内容', '客户端投诉问题分类', '投诉问题定位']
# KEYS = ['受理内容', '投诉问题定位']
SPECIAL_KEYS = ['serv_type']
PRE_TOKENS_LABEL = {'<START>': 0,
                    '<END>': 1,
                    '<PAD>': 2}