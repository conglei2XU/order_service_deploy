export CUDA_VISIBLE_DEVICES=-1
gunicorn -c service_config/gun_config.py run:app
