# PyRat ReinforcementLearning (PyTorch)

This code uses an approach of Policy Gradients and REINFORCE for training a neural network capable to chose best actions for win the PyRat game against a Greedy Algorithm. It can win the Greedy Algorithm 83.0% of games in the configuration described by the follows command lines.

The important files for this AI are:
- AIs/ratorch.py
- AIs/ratorch_test.py 

## For training
```
python3 ./pyrat.py --python AIs/cours5.py --rat AIs/ratorch.py --random_seed 2017 -p 100 --tests 100000 --random_cheese -d 0.1 -md 0 --nodrawing
```

## For testing
```
python3 ./pyrat.py --python AIs/cours5.py --rat AIs/ratorch_test.py --random_seed 2017 -p 100 --tests 100 --random_cheese -d 0.1 -md 0
```
The are checkpoints of a training network are also available:

*** checkpoint.pth.tar***




The PyRat original game the credits are available in https://github.com/vgripon. Please, follow the instructions in the same link for install the game before run the Reiforcement Learning IA.
