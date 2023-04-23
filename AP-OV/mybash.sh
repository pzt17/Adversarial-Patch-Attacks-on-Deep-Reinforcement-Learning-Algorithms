#! /bin/bash

python patch_test.py --env Freeway --percent_image 0.0005
python patch_test.py --env Freeway --percent_image 0.001
python patch_test.py --env Freeway --percent_image 0.003
python patch_test.py --env Freeway --percent_image 0.005
python patch_test.py --env Freeway --percent_image 0.01
#python patch_train.py --env Freeway --percent_image 0.03
#python patch_train.py --env Freeway --percent_image 0.05

python patch_test.py --env BankHeist --percent_image 0.0005
python patch_test.py --env BankHeist --percent_image 0.001
python patch_test.py --env BankHeist --percent_image 0.003
python patch_test.py --env BankHeist --percent_image 0.005
python patch_test.py --env BankHeist --percent_image 0.01
#python patch_train.py --env BankHeist --percent_image 0.03
#python patch_train.py --env BankHeist --percent_image 0.05

python patch_test.py --env RoadRunner --percent_image 0.0005
python patch_test.py --env RoadRunner --percent_image 0.001
python patch_test.py --env RoadRunner --percent_image 0.003
python patch_test.py --env RoadRunner --percent_image 0.005
python patch_test.py --env RoadRunner --percent_image 0.01
#python patch_train.py --env RoadRunner --percent_image 0.03
#python patch_train.py --env RoadRunner --percent_image 0.05

python patch_test.py --env Qbert --percent_image 0.0005
python patch_test.py --env Qbert --percent_image 0.001
python patch_test.py --env Qbert --percent_image 0.003
python patch_test.py --env Qbert --percent_image 0.005
python patch_test.py --env Qbert --percent_image 0.01
#python patch_train.py --env Qbert --percent_image 0.03
#python patch_train.py --env Qbert --percent_image 0.05

python patch_test.py --env Pong --percent_image 0.0005
python patch_test.py --env Pong --percent_image 0.001
python patch_test.py --env Pong --percent_image 0.003
python patch_test.py --env Pong --percent_image 0.005
python patch_test.py --env Pong --percent_image 0.01
#python patch_train.py --env Pong --percent_image 0.03
#python patch_train.py --env Pong --percent_image 0.05
