# EMGAIL

This code_appendix is the implementation of the paper:

**Model-Free Inverse Reinforcement Learning with Multi-Intention, Incomplete, and Overlapping Demonstrations</a>**


## Dependencies
The code is developed and tested on Ubuntu 18.04 with Python 3.6 and PyTorch 1.9.

You can install all the dependencies by running:

```bash
pip install -r requirements.txt # Install dependencies
```

Implementation of "Deep Adaptive Multi-intention Inverse Reinforcement Learning"

## Experiments
For the sake of simplicity, we provide all the experimental details of the "swimmer" environment.

### Multi-intention version
The default swimmer in the gym, is a single-intention environment. We modified the file "swimmer.py" to be a two-intention environment and it should be replaced in the gym library.

### Reinforcement learning (RL)
For each experiment, we first obtain the expert's policy by running RL on the true reward functions to generate the expert's demonstrations. The RL policies for both intentions are already trained and saved in the "checkpoints" file.

To test the RL policy run:
```bash
python3 rl/main.py --test-rl=True --intention-idx=0 # intention-idx could be either 0 or 1
```

To generate the demonstrations run:
```bash
python3 rl/main.py --generate-demons=True --intention-idx=0 # intention-idx could be either 0 or 1
```

To train the RL policy from scratch run:
```bash
python3 rl/main.py --train-rl=True --intention-idx=0 # intention-idx could be either 0 or 1
```

### Inverese Reinforcement learning (IRL)
Given the expert's demonstrations, the the IRL algorithm (EMGAIL) can be run. The multi-intention IRL policy is already trained and saved in the "checkpoints" file. Please note that the order of true intentions in RL and the learned intentions in IRL may be different. 

To test the multi-intention IRL policy run:
```bash
python3 irl/main.py --test-irl=True --test-intention=0 # test-intention could be either 0 or 1
```

To train the multi-intention IRL policy from scratch run:
```bash
python3 irl/main.py --train-irl=True
```
