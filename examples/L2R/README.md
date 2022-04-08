# Learning-to-reweight-examples

Code On Mnist reweighting example in paper [Learning to Reweight Examples for Robust Deep Learning](https://arxiv.org/abs/1803.09050)] using `TorchOpt`. The idea of L2R is to use virtual update of inner-loop neural network optimisation to meta-learn the reweighting parameters for robust deep learning. We use `MetaSGD` as the inner-loop optimiser.

# Usage
We use traditional supervised training as the baseline.
```bash
### Run both algorithms and conduct comparison
python3 train_l2r.py --algo both

### For baseline 
python3 train_l2r.py --algo baseline

### For L2R algorithm
python3 train_l2r.py --algo l2r
```

# Results
The test accuracy comparison between baseline and L2R validate the effectiveness of algorithms.
<div align=center>
<img src="./result.png" width="450" height="325" />
</div>

