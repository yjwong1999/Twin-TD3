# Deep Reinforcement Learning for Secrecy Energy-Efficient UAV Communication with Reconfigurable Intelligent Surfaces

**IEEE Wireless Communications and Networking Conference 2023 (WCNC 2023)** </br>
Simulation for Conference Proceedings "doi-to-be-filled" </br>
Refer [this link](https://github.com/yjwong1999/Twin-TD3/blob/main/WCNC2023%20WS-09%20%231570879488.pdf) for the preprint

## Abstract
This paper investigates the **physical layer security (PLS)** issue in **reconfigurable intelligent surface (RIS) aided millimeter-wave rotary-wing unmanned aerial vehicle (UAV) communications** under the presence of multiple eavesdroppers and imperfect channel state information (CSI). The goal is to maximize the **worst-case secrecy energy efficiency (SEE)** of UAV via a **joint optimization of flight trajectory, UAV active beamforming and RIS passive beamforming**. By interacting with the dynamically changing UAV environment, real-time decision making per time slot is possible via deep reinforcement learning (DRL). To decouple the continuous optimization variables, we introduce a **twin twin-delayed deep deterministic policy gradient (TTD3)** to maximize the expected cumulative reward, which is linked to SEE enhancement. Simulation results confirm that the proposed method achieves greater secrecy energy savings than the traditional twin-deep deterministic policy gradient DRL (TDDRL)-based method. 

## TLDR

### System model: 
**RIS-aided mmWave UAV system** under the presence of **eavesdroppers** and **imperfect channel state information (CSI)** </br>

### Solution: 
A **Twin-TD3 (TTD3) algorithm** to decouple the joint optimization of:
1. UAV active beamforming and RIS passive beamforming 
2. UAV flight trajectory

We adopt **double DRL framework**, where the 1st and 2nd agent provides the policy for task (1) and (2), respectively.

## How to use this repo
User can train two types of algorithm for training:
1. Twin DDPG is [TDDRL algorithm](https://doi.org/10.1109/LWC.2021.3081464)
2. Twin TD3 is our proposed TTD3 algorithm

Run the following  in the `bash` or `powershell`

`main_test.py` is the main python file to train the DRL algorithms
```shell
# To use Twin DDPG with SSR as optimization goal
python3 main_test.py --drl ddpg --reward ssr

# To use Twin TD3 with SSR as optimization goal
python3 main_test.py --drl td3 --reward ssr

# To use Twin DDPG with SEE as optimization goal
python3 main_test.py --drl ddpg --reward see

# To use Twin TD3 with SEE as optimization goal
python3 main_test.py --drl td3 --reward see



# To use pretrained DRL for UAV trajectory (recommended for stable convergence)
python3 main_test.py --drl td3 --reward see --trained_uav

# To set number of episodes (default is 300)
python3 main_test.py --drl td3 --reward see --ep_num 300

# To set seeds for DRL weight initialization (not recommended if you use pretrained uav)
python3 main_test.py --drl td3 --reward see --seeds 0       # weights of both DRL are initialized with seed 0
python3 main_test.py --drl td3 --reward see --seeds 0 1     # weights of DRL 1 and DRL2 are initialized with seed 0 and 1, respectively
```

`load_and_plot.py` is the python file to plot the (i) Rewards, (ii) Sum Secrecy Rate (SSR), (iii) Secrecy Energy Efficient (SEE), (iv) UAV Trajectory, (v) RIS configs for each episode in one experiments
```shell
# plot everything for each episode
python3 load_and_plot.py --path data/storage/<DIR> --ep_num 300
```

`plot_ssr.py` is the python file to plot the final episode's SSR for the 4 benchmarks in the paper
```shell
# plot ssr
python3 plot_ssr.py
```

`plot_see.py` is the python file to plot the final episode's SSR for the 4 benchmarks in the paper
```shell
# plot see
python3 plot_see.py
```

`plot_traj.py` is the python file to plot the final episode's UAV trajectory for the 4 benchmarks in the paper
```shell
# plot UAV trajectory
python3 plot_traj.py
```

## Results

We run the ```main_test.py``` for 5 times for each settings below, and averaged out the performance

SSR and SEE              (the higher the better)
Total Energy Consumption (the lower the better)

| Algorithms                     | SSR (bits/s/Hz)| Energy (kJ) | SEE (bits/s/Hz/kJ)|
|--------------------------------|----------------|-------------|-------------------|
| TDDRL                          | 5.03           | 12.4        | 40.8              |
| TTD3                           | 6.05           | 12.7        | 48.2              |
| TDDRL (with energy constraint) | 4.68           | 11.2        | 39.4              |
| TTD3  (with energy constraint) | 5.39           | 11.2        | 48.4              |

Summary
1. In terms of SSR, TTD3 outperforms TDDRL with or without energy constraint
2. In terms of SEE and Energy, TTD3 (with energy constraint) outperforms all other algorithms
3. Generally, TTD3 algorithm are better than TTDRL
4. Even with energy contraint (trade-off between energy consumption and SSR), TTD3 outperforms TDDRL in all aspects

\* Remarks: </br>
Note that the performance of DRL (especially twin DRL) has a big variation, sometimes you may get extremely good (or bad) performance </br>
It is advised to use the benchmark UAV models we trained, for better convergence. </br>
This approach is consistent with the codes provided by [TDDRL](https://github.com/Brook1711/WCL-pulish-code)

## References and Acknowledgement

This work was supported by the **British Council** under **UK-ASEAN Institutional Links Early Career Researchers Scheme** with project number 913030644.

Both **RIS Simulation** and the **System Model** for this Research Project are based the research work provided by [Brook1711](https://github.com/Brook1711). </br>
We intended to fork the original repo for the system model (as stated below) as the base of this project. </br>
However, GitHub does not allow a forked repo to be private. </br>
Hence, we could not maintain our code based a forked version of the original repo, while keeping it private until the project is completed.
We would like to express our utmost gratitude for [Brook1711](https://github.com/Brook1711) and his co-authors for their research work.

### RIS Simulation
RIS Simulation is based on the following research work: </br>
[SimRIS Channel Simulator for Reconfigurable Intelligent Surface-Empowered Communication Systems](https://ieeexplore.ieee.org/document/9282349) </br>
The original simulation code is coded in matlab, this [GitHub repo](https://github.com/Brook1711/RIS_components) provides a Python version of the simulation.

### System Model: RIS-aided mmWave UAV communications
The simulation of the System Model is provided by the following research work: </br>
[Learning-Based Robust and Secure Transmission for Reconfigurable Intelligent Surface Aided Millimeter Wave UAV Communications](https://doi.org/10.1109/LWC.2021.3081464) </br>
The code is provided in this [GitHub repo](https://github.com/Brook1711/WCL-pulish-code).

### Rotary-Wing UAV
We can derive the Rotary-Wing UAV’s propulsion energy consumption based on the following research work: </br>
[Energy Minimization in Internet-of-Things System Based on Rotary-Wing UAV](https://doi.org/10.1109/LWC.2019.2916549)

### TD3
Main reference for TD3 implementation: </br>
[PyTorch/TensorFlow 2.0 for TD3](https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/tree/master/TD3)


## TODO
- [x] Add argparse arguments to set drl algo and reward type
- [x] Add argparse arguments to set episode number
- [x] Add argparse arguments to set seeds for the two DRLs
- [x] Add argparse arguments to load pretrained DRL for UAV trajectory
- [x] Add benchmark/pretrained model
- [ ] Remove saving "best model"
