<h1 align="center"> Deep Autoregressive Models </h1>
<p align="center">
<img height="250" src="https://raw.githubusercontent.com/Mind-the-Pineapple/Autoregressive-models/master/1%20-%20Autoregressive%20Models%20-%20PixelCNN/figures/Figure6_reduced.png"> 
</p>

[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](https://github.com/Warvito/vq-vae/blob/master/LICENSE)

This repository is a supplement to our blog series *Deep Autoregressive Models*.

## Setup
Clone the git repository :

    git clone https://github.com/Mind-the-Pineapple/Autoregressive-models.git

Python 3 with [TensorFlow 2.0+](https://www.tensorflow.org/) are the primary requirements.
Install virtualenv and creat a new virtual environment:

    sudo apt-get install -y python3-venv
    python3 -m venv ./venv

Then, install requirements

    source venv/bin/activate
    pip3 install -r requirements.txt

<h1 align="center"> 1. Autoregressive Models — PixelCNN </h1>
<img align="right" width="500x" src="https://raw.githubusercontent.com/Mind-the-Pineapple/Autoregressive-models/master/1%20-%20Autoregressive%20Models%20-%20PixelCNN/figures/Figure5_Architecture_reduced.png">

Creating digits with deep generative models!
- [PixelCNN Medium Story](https://towardsdatascience.com/autoregressive-models-pixelcnn-e30734ede0c1)
- [Google Colab](https://colab.research.google.com/github/Mind-the-Pineapple/Autoregressive-models/blob/master/1%20-%20Autoregressive%20Models%20-%20PixelCNN/pixelCNN.ipynb)
- [Code](https://github.com/Mind-the-Pineapple/Autoregressive-models/blob/master/1%20-%20Autoregressive%20Models%20-%20PixelCNN/pixelCNN.py)
- [Paper -> Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759)
