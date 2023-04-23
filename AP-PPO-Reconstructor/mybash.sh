#! /bin/bash

python main.py --env_name coinrun --model-file trained_models/CoinRun_ppo.pt --percent_image 0.05
python main.py --env_name fruitbot --model-file trained_models/FruitBot_ppo.pt --percent_image 0.05
python main.py --env_name jumper --model-file trained_models/Jumper_ppo.pt --percent_image 0.05
