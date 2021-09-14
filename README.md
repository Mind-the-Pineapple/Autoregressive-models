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
Install virtualenv and create a new virtual environment:

    sudo apt update
    sudo apt install python3-dev python3-pip
    sudo pip3 install -U virtualenv  # system-wide install
    virtualenv --system-site-packages -p python3 ./venv

Then, install requirements

    source ./venv/bin/activate
    pip3 install --upgrade pip
    pip3 install -r requirements.txt

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
<h1 align="center"> 1. Autoregressive Models — PixelCNN </h1>
<img align="right" width="500x" src="https://raw.githubusercontent.com/Mind-the-Pineapple/Autoregressive-models/master/1%20-%20Autoregressive%20Models%20-%20PixelCNN/figures/Figure5_Architecture_reduced.png">

Creating digits with deep generative models!
- [PixelCNN Medium Story](https://towardsdatascience.com/autoregressive-models-pixelcnn-e30734ede0c1)
- [Google Colab](https://colab.research.google.com/github/Mind-the-Pineapple/Autoregressive-models/blob/master/1%20-%20Autoregressive%20Models%20-%20PixelCNN/pixelCNN.ipynb)
- [Code](https://github.com/Mind-the-Pineapple/Autoregressive-models/blob/master/1%20-%20Autoregressive%20Models%20-%20PixelCNN/pixelCNN.py)
- [Paper -> Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759)


<br>
<br>
<h1 align="center"> 2. Modelling Coloured Images </h1>
<img align="right" width="500x" src="https://raw.githubusercontent.com/Mind-the-Pineapple/Autoregressive-models/master/2%20-%20Modelling%20data%20with%20multiple%20channels/figures/PixelcnnRGB.png">


Extending the model to the full colour spectrum
- [PixelCNN - RGB Medium Story](https://blog.usejournal.com/modelling-coloured-images-acd0ebde0102)
- [Google Colab](https://colab.research.google.com/gist/PedroFerreiradaCosta/a770317efa23f36b3c5009a9f21169f3/pixelcnn-rgb.ipynb?authuser=1#scrollTo=deJgSHmOBGfk)
- [Code](https://github.com/Mind-the-Pineapple/Autoregressive-models/blob/master/2%20-%20Modelling%20data%20with%20multiple%20channels/pixelCNN_RGB.py)
- [Paper -> Pixel Recurrddent Neural Networks](https://arxiv.org/abs/1601.06759)

<br>
<br>
<br>
<h1 align="center"> 3. PixelCNN's Blind Spot and how to fix it </h1>
<img align="right" width="500x" src="https://raw.githubusercontent.com/Mind-the-Pineapple/Autoregressive-models/master/3%20-%20PixelCNNs%20Blind%20spot%20and%20Gated%20PixelCNNs/figures/results.png">

How to fix the Pixel CNN's blind spot
- [PixelCNN Medium Story](https://towardsdatascience.com/pixelcnns-blind-spot-84e19a3797b9)
- [Google Colab](https://colab.research.google.com/github/Mind-the-Pineapple/Autoregressive-models/blob/master/3%20-%20PixelCNNs%20Blind%20spot%20and%20Gated%20PixelCNNs/PixelCNN_Blind_spot.ipynb)
- [Code](https://github.com/Mind-the-Pineapple/Autoregressive-models/blob/master/3%20-%20PixelCNNs%20Blind%20spot%20and%20Gated%20PixelCNNs/pixelcnn_blind_spot.py)
- [Paper -> Gated PixelCNNs](https://arxiv.org/abs/1606.05328)

