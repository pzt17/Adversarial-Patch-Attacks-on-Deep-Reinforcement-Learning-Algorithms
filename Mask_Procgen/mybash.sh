#! /bin/bash

python mask_train.py --env_name coinrun --model_file trained_models/CoinRun_ppo.pt --percent_image 0.005 
python mask_train.py --env_name fruitbot --model_file trained_models/FruitBot_ppo.pt --percent_image 0.005  --train_episodes 300000
python mask_train.py --env_name jumper --model_file trained_models/Jumper_ppo.pt --percent_image 0.005  

'''
python mask_train.py --env_name coinrun --model_file trained_models/CoinRun_ppo.pt --percent_image 0.01
python mask_train.py --env_name fruitbot --model_file trained_models/FruitBot_ppo.pt --percent_image 0.01 --train_episodes 300000
python mask_train.py --env_name jumper --model_file trained_models/Jumper_ppo.pt --percent_image 0.01

python mask_train.py --env_name coinrun --model_file trained_models/CoinRun_ppo.pt --percent_image 0.03
python mask_train.py --env_name fruitbot --model_file trained_models/FruitBot_ppo.pt --percent_image 0.03 --train_episodes 300000
python mask_train.py --env_name jumper --model_file trained_models/Jumper_ppo.pt --percent_image 0.03

python mask_train.py --env_name coinrun --model_file trained_models/CoinRun_ppo.pt --percent_image 0.05
python mask_train.py --env_name fruitbot --model_file trained_models/FruitBot_ppo.pt --percent_image 0.05 --train_episodes 300000
python mask_train.py --env_name jumper --model_file trained_models/Jumper_ppo.pt --percent_image 0.05
'''
