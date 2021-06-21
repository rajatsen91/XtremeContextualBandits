# eXtreme Contextual Bandits
Code for Top-k eXtreme Contextual Bandits: https://arxiv.org/abs/2102.07800

##Data

We include utils to process the datasets in the XMC repository (https://tinyurl.com/4m7eczdv) to our input format. Download a dataset from the original link for instance Eurlex-4k. 

Then run the following command:

```shell
python xcb.utils.convert2sparse -i eurlex_train.txt -o path/to/train --normalize

```

The ```normalize``` flag normalizes the features to unit l2-norm and is quite important for our experiments. 

##Installation

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

##Usage




# Citation
Please cite https://arxiv.org/abs/2102.07800 if using this code for a publication.
