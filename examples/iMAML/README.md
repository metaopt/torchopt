# implicit MAML few-shot Omniglot classification-examples

Code on implicit MAML few-shot Omniglot classification in paper [Meta-learning with implicit gradients](https://arxiv.org/abs/1909.04630) using TorchOpt. We use `MetaSGD` as the inner-loop optimizer.

## Usage

```bash
### Run
python3 imaml_omniglot.py --inner_steps 5
```

## Results

The figure illustrate the experimental result.

<div align=center>
  <img src="./imaml-accs.png" width="450" height="325" />
</div>
