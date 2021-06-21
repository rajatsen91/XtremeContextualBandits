# eXtreme Contextual Bandits
Code for Top-k eXtreme Contextual Bandits: https://arxiv.org/abs/2102.07800

## Data

We include utils to process the datasets in the XMC repository (https://tinyurl.com/4m7eczdv) to our input format. Download a dataset from the original link for instance Eurlex-4k. 

Then run the following command:

```shell
python xcb.utils.convert2sparse -i eurlex_train.txt -o path/to/train --normalize

```

The ```normalize``` flag normalizes the features to unit l2-norm and is quite important for our experiments. 

## Installation

Clone the package ```recursively```.

```shell
git clone --recursive https://github.com/rajatsen91/XtremeContextualBandits.git

```

Install the requirements.

```shell
pip install -r requirements.txt

```

Install the package by navigating to the base directory.

```shell
pip install -e .

```

## Usage

An example code snippet for running an experiment is given below:

```python
import os
from xcb.eval import simulator as simulator
from xcb.xfalcon.inference import XLinearCBI
from xcb.xfalcon.train import XFalconTrainer
import scipy.sparse as smat
import numpy as np
from xcb.utils.logging import setup_logging_config
import scipy as sp
import logging
import warnings

LOGGER = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
setup_logging_config()


topk = 5
num_explore = 3
max_leaf_size = 100
beam_size = 10

Xtrain = smat.load_npz("path/train/X.npz")
Ytrain = smat.load_npz("path/train/Y.npz")
Xtest = smat.load_npz("path/test/X.npz")
Ytest = smat.load_npz("path/test/Y.npz")
tei = np.random.permutation(np.arange(Xtest.shape[0]))
tri = np.random.choice(Xtrain.shape[0], Xtrain.shape[0])
# Size of initialization supervised set used for training tree and routing functions
init_batch = 5000
Xtrain, Ytrain = Xtrain[tri, :], Ytrain[tri, :]
Xtest, Ytest = Xtest[tei[0:init_batch], :], Ytest[tei[0:init_batch], :]

init_args = {"X": Xtest, "Y": Ytest}
eval_args = {
    "X": Xtrain, "Y": Ytrain, "batch_size": init_batch, "schedule": "exponential",
    "topk": topk
}
train_config = {
    "model_config": {
        "threshold": 0.1,
        "learner": {"class": "SVC", "reg":"SVR"},
        "linear_config": {"class": {"tol": 0.1, "max_iter" : 40}, "reg": {"tol": 0.1, "max_iter" : 40}}
    },
    "cluster_config": {"max_leaf_size": max_leaf_size},
    "mode": "ranker"
}

inference_config = {
    "mode": "falcon",
    "pred_config": {
        "beam_size": beam_size,
        "topk": topk,
        "post_processor":"l3-hinge",
        "combiner": "multiply",
        "explore_in_routing": False,
        "multiplier": 1.0,
        "num_explore": num_explore,
        "explore_strategy": "falcon",
        "alpha": 0.5,
    }
}

LOGGER.info("Inference Config: {}".format(inference_config))
sim = simulator.XCBSimulatedBandit(init_args, eval_args, train_config, inference_config)
rewards_falcon = sim.evaluate()  # this stores the collected reward per time-step. Can be normalized to yield progressive mean reward/ loss.

```




# Citation
Please cite https://arxiv.org/abs/2102.07800 if using this code for a publication.
