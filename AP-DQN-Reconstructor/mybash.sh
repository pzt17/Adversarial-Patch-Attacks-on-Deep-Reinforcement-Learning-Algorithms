#! /bin/bash

python train_base.py --env FreewayNoFrameskip-v4 --load_model_path Freeway.pt 
python train_base.py --env QbertNoFrameskip-v4 --load_model_path Qbert.pt --num_frames 30000
python train_base.py --env BankHeistNoFrameskip-v4 --load_model_path BankHeist.pt  --num_frames 20000
python train_base.py --env PongNoFrameskip-v4 --load_model_path Pong.pt --num_frames 20000
python train_base.py --env RoadRunnerNoFrameskip-v4 --load_model_path RoadRunner.pt  --num_frames 30000
