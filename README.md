# Deep Reinforcement Learning for Secrecy Energy-Efficient UAV Communication with Reconfigurable Intelligent Surfaces

**IEEE Wireless Communications and Networking Conference 2023 (WCNC 2023)** </br>
Simulation for Conference Proceedings "doi-to-be-filled"

Note that the `main_test.py` is the main running python file. To run it, please type 
```shell
python main_test.py
python3 main_test.py
```
in the `bash` or `powershell`


## References and Acknowledgement

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
[Learning-Based Robust and Secure Transmission for Reconfigurable Intelligent Surface Aided Millimeter Wave UAV Communications]() </br>
The code is provided in this [GitHub repo](https://github.com/Brook1711/WCL-pulish-code).

### Rotary-Wing UAV
We can derive the Rotary-Wing UAV’s propulsion energy consumption based on the following research work: </br>
[Energy Minimization in Internet-of-Things System Based on Rotary-Wing UAV](https://doi.org/10.1109/LWC.2019.2916549)

### TD3
Main reference for TD3 implementation: </br>
[PyTorch/TensorFlow 2.0 for TD3](https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/tree/master/TD3)
