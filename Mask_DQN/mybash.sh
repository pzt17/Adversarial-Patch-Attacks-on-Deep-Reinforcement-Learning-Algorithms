#! /bin/bash

#python mask_train.py --env QbertNoFrameskip-v4 --load_model_path Qbert.pt --percent_image 0.003
#python mask_train.py --env QbertNoFrameskip-v4 --load_model_path Qbert.pt --percent_image 0.01
#python mask_train.py --env QbertNoFrameskip-v4 --load_model_path Qbert.pt --percent_image 0.03
#python mask_train.py --env QbertNoFrameskip-v4 --load_model_path Qbert.pt --percent_image 0.05

#python mask_train.py --env BankHeistNoFrameskip-v4 --load_model_path BankHeist.pt --percent_image 0.003
#python mask_train.py --env BankHeistNoFrameskip-v4 --load_model_path BankHeist.pt --percent_image 0.01
#python mask_train.py --env BankHeistNoFrameskip-v4 --load_model_path BankHeist.pt --percent_image 0.03
#python mask_train.py --env BankHeistNoFrameskip-v4 --load_model_path BankHeist.pt --percent_image 0.05

python mask_train.py --env RoadRunnerNoFrameskip-v4 --load_model_path RoadRunner.pt --percent_image 0.003 
#python mask_train.py --env RoadRunnerNoFrameskip-v4 --load_model_path RoadRunner.pt --percent_image 0.01
#python mask_train.py --env RoadRunnerNoFrameskip-v4 --load_model_path RoadRunner.pt --percent_image 0.03  
#python mask_train.py --env RoadRunnerNoFrameskip-v4 --load_model_path RoadRunner.pt --percent_image 0.05 

#python mask_train.py --env FreewayNoFrameskip-v4 --load_model_path Freeway.pt --percent_image 0.003
#python mask_train.py --env FreewayNoFrameskip-v4 --load_model_path Freeway.pt --percent_image 0.01
#python mask_train.py --env FreewayNoFrameskip-v4 --load_model_path Freeway.pt --percent_image 0.03
#python mask_train.py --env FreewayNoFrameskip-v4 --load_model_path Freeway.pt --percent_image 0.05

python mask_train.py --env PongNoFrameskip-v4 --load_model_path Pong.pt --percent_image 0.003 --epsilon 0.3
#python mask_train.py --env PongNoFrameskip-v4 --load_model_path Pong.pt --percent_image 0.01 --epsilon 0.3
#python mask_train.py --env PongNoFrameskip-v4 --load_model_path Pong.pt --percent_image 0.03 --epsilon 0.3
#python mask_train.py --env PongNoFrameskip-v4 --load_model_path Pong.pt --percent_image 0.005 --epsilon 0.3
