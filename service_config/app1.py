import eventlet
eventlet.monkey_patch()

# 绑定ip和端口号
bind = '0.0.0.0:10082'
# 进程数量
workers = 1
# 使用eventlet模式
worker_class = 'eventlet'
max_requests = 1
max_requests_jitter = 2
# 响应耗时
timeout = 200
graceful_timeout = 30

limit_request_line = 4094
limit_request_fields = 100
limit_request_fields_size = 8190

loglevel = 'debug'
capture_output = True
accesslog = 'service_log/app1.log'
errorlog = 'service_log/app1.error.log'
