#!/bin/sh
export CUDA_VISIBLE_DEVICES=-1
service=("app1" "app2" "app3")

port=10081
gunicorn -c service_config/app1.py app1:app & \
gunicorn -c service_config/app2.py app2:app & \
gunicorn -c service_config/app3.py app3:app & \
gunicorn -c service_config/main_app.py run:app
# for item in "${service[@]}"; do
#     port=$(( ${port} + 1 ))
#     # flask --app ${item} run --debug --host 0.0.0.0:"${port}"
#     gunicorn -c service_config/"${item}".py "${item}":app &
#     echo "${item} ${port}"
# done
# gunicorn -c service_config/"${item}".py "${item}":app &
# gunicorn -c service_config/main_app.py run:app
#  
# wait 


# array=("item 1" "item 2" "item 3")
# for i in "${array[@]}"; do   # The quotes are necessary here
#     echo "$i"
# done