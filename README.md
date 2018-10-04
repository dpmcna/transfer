### Code and data for experiments from 'Risk Bounds for Transferring Representations With and Without Fine-Tuning', Daniel McNamara and Maria-Florina Balcan, International Conference on Machine Learning, 2017.

The paper is viewable [here](http://proceedings.mlr.press/v70/mcnamara17a/mcnamara17a.pdf). The code was developed by Daniel McNamara.

The experiments were run in MATLAB. For the MNIST experiments, run `run_mnist_experiments.m`. For the NEWSGROUPS experiments, run `run_newsgroups_experiments.m`. Both sets of experiments are runnable on a regular desktop but may take several hours to a few days to run. In the paper we ran each set of experiments 10 times and averaged the results, but you can also more quickly run each experiment just once.

We obtained the MNIST and NEWSGROUPS datasets from http://yann.lecun.com/exdb/mnist and http://qwone.com/~jason/20Newsgroups respectively. Some code was adapted from [Andrew Ng's Coursera Machine Learning course](http://www.coursera.org/learn/machine-learning).