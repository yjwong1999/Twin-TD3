#!/bin/bash

echo 

echo ddpg_ssr
python3 main_test.py --drl ddpg --reward ssr --ep_num 300 --trained_uav
echo ddpg_ssr_2
python3 main_test.py --drl ddpg --reward ssr --ep_num 300 --trained_uav
echo ddpg_ssr_3
python3 main_test.py --drl ddpg --reward ssr --ep_num 300 --trained_uav
echo ddpg_ssr_4
python3 main_test.py --drl ddpg --reward ssr --ep_num 300 --trained_uav
echo ddpg_ssr_5
python3 main_test.py --drl ddpg --reward ssr --ep_num 300 --trained_uav

echo td3_ssr
python3 main_test.py --drl td3 --reward ssr --ep_num 300 --trained_uav
echo td3_ssr_2
python3 main_test.py --drl td3 --reward ssr --ep_num 300 --trained_uav
echo td3_ssr_3
python3 main_test.py --drl td3 --reward ssr --ep_num 300 --trained_uav
echo td3_ssr_4
python3 main_test.py --drl td3 --reward ssr --ep_num 300 --trained_uav
echo td3_ssr_5
python3 main_test.py --drl td3 --reward ssr --ep_num 300 --trained_uav

echo ddpg_see
python3 main_test.py --drl ddpg --reward see --ep_num 300 --trained_uav
echo ddpg_see_2
python3 main_test.py --drl ddpg --reward see --ep_num 300 --trained_uav
echo ddpg_see_3
python3 main_test.py --drl ddpg --reward see --ep_num 300 --trained_uav
echo ddpg_see_4
python3 main_test.py --drl ddpg --reward see --ep_num 300 --trained_uav
echo ddpg_see_5
python3 main_test.py --drl ddpg --reward see --ep_num 300 --trained_uav

echo td3_see
python3 main_test.py --drl td3 --reward see --ep_num 300 --trained_uav
echo td3_see_2
python3 main_test.py --drl td3 --reward see --ep_num 300 --trained_uav
echo td3_see_3
python3 main_test.py --drl td3 --reward see --ep_num 300 --trained_uav
echo td3_see_4
python3 main_test.py --drl td3 --reward see --ep_num 300 --trained_uav
echo td3_see_5
python3 main_test.py --drl td3 --reward see --ep_num 300 --trained_uav
