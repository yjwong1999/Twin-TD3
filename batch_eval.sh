#!/bin/bash

echo 

echo ddpg_ssr
python3 load_and_plot.py --path data/storage/ddpg_ssr --ep-num 300
echo ddpg_ssr_2
python3 load_and_plot.py --path data/storage/ddpg_ssr_2 --ep-num 300
echo ddpg_ssr_3
python3 load_and_plot.py --path data/storage/ddpg_ssr_3 --ep-num 300
echo ddpg_ssr_4
python3 load_and_plot.py --path data/storage/ddpg_ssr_4 --ep-num 300
echo ddpg_ssr_5
python3 load_and_plot.py --path data/storage/ddpg_ssr_5 --ep-num 300


echo td3_ssr
python3 load_and_plot.py --path data/storage/td3_ssr --ep-num 300
echo td3_ssr_2
python3 load_and_plot.py --path data/storage/td3_ssr_2 --ep-num 300
echo td3_ssr_3
python3 load_and_plot.py --path data/storage/td3_ssr_3 --ep-num 300
echo td3_ssr_4
python3 load_and_plot.py --path data/storage/td3_ssr_4 --ep-num 300
echo td3_ssr_5
python3 load_and_plot.py --path data/storage/td3_ssr_5 --ep-num 300


echo ddpg_see
python3 load_and_plot.py --path data/storage/ddpg_see --ep-num 300
echo ddpg_see_2
python3 load_and_plot.py --path data/storage/ddpg_see_2 --ep-num 300
echo ddpg_see_3
python3 load_and_plot.py --path data/storage/ddpg_see_3 --ep-num 300
echo ddpg_see_4
python3 load_and_plot.py --path data/storage/ddpg_see_4 --ep-num 300
echo ddpg_see_5
python3 load_and_plot.py --path data/storage/ddpg_see_5 --ep-num 300


echo td3_see
python3 load_and_plot.py --path data/storage/td3_see --ep-num 300
echo td3_see_2
python3 load_and_plot.py --path data/storage/td3_see_2 --ep-num 300
echo td3_see_3
python3 load_and_plot.py --path data/storage/td3_see_3 --ep-num 300
echo td3_see_4
python3 load_and_plot.py --path data/storage/td3_see_4 --ep-num 300
echo td3_see_5
python3 load_and_plot.py --path data/storage/td3_see_5 --ep-num 300


