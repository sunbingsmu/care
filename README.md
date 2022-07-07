# care

This is the implementation of CARE (causality based neural network repair) (https://doi.org/10.48550/arXiv.2204.09274).

For fairness improvement task, please refer to https://github.com/longph1989/Socrates.
For backdoor removal and safety improvement task, please run below script in each module:
- causal_analysis.py 

Benchmark is available at [benchmark](https://figshare.com/articles/dataset/CARE_benchmark_v2/20250816).

We are migrating this implementation to Socrates and this is a temparary repo for now.

Experiments:
- NN4: root causal_analysis.py
- NN5: mnist folder
- NN6: fashion folder
- NN7: acas_N29 folder
- NN8: acas_N33 folder
- NN9: acas_N19 folder
- mnist_nnrepair: mnist model to compare with nnrepair
- cifar_nnrepair: cifar model to compare with nnrepair
