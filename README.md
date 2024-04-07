# super_policy_learning
This is the code accompanying with the paper Wang et. al. (2022). The code is constructed based on the files from https://github.com/rui-miao/ProxITR. 

# file description:
**`single_stage_tabular_run.py`** and **`single_stage_continous_run.py`** are for runing simulations described in the paper for contextual bandits. 

**`multi_stage_longterm_run.py`** and **`multi_stage_DTR_run.py`** are for runing simulations described in the paper for sequential decision making. 

# examples:
```
python single_stage_continuous_run.py --n=1000 --epsilon=0.9 --seed=1
python single_stage_tabular_run.py --n=5000 --epsilon=0.9 --seed=8

python multi_stage_longterm_run.py --n=2000 --T=20 --seed=8
python multi_stage_DTR_run.py --n=2000 --T=2 --seed=8
```

## References
* Wang, Jiayi, Zhengling Qi, and Chengchun Shi. "Blessing from experts: Super reinforcement learning in confounded environments." arXiv preprint arXiv:2209.15448 (2022).
