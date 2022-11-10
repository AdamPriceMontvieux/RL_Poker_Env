Spin up EC2 
t3.2xl
SSH only from MV AWS VPN


sudo apt update 
sudo apt install python3-pip

pip install --upgrade -r requirements.txt

python3 -m notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root --NotebookApp.iopub_data_rate_limit=1000000 --NotebookApp.token='' 


python3 -m tensorboard.main --logdir=~/ray_results --host 0.0.0.0


Look at 
http://localhost:6006/?tagFilter=episode_reward_mean#timeseries 
Filter on episode_reward_mean 
