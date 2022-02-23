# EK-UCB

Efficient-Kernel UCB implementation

To install the needed libraries, run

```
pip install -r requirements.txt
```

To run an experiment, run according to the following examples:

EK-UCB algorithm on the bump environment with parameters lambda, mu, beta (KORS)
```
python run.py --algo ek_ucb --env bump --mu 1 --lambda 1 --beta 0.1 
```

SupK-UCB algorithm on the squares environment with parameters lambda
```
python run.py --algo supk_ucb --env squares --lambda 1

```

See the file run.py for detailed commands.

To run the full experiments, run 

```
bash script.sh
```

