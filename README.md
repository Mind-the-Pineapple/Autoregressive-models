# Neural Discrete Representation Learning
Tensorflow 2.0 implementation of the Vector Quantised-Variational AutoEncoder

[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](https://github.com/Warvito/vq-vae/blob/master/LICENSE)

## Abstract
Learning useful representations without supervision remains a key challenge in machine learning. In this paper, we propose a simple yet powerful generative model that learns such discrete representations. Our model, the Vector Quantised-Variational AutoEncoder (VQ-VAE), differs from VAEs in two key ways: the encoder network outputs discrete, rather than continuous, codes; and the prior is learnt rather than static. In order to learn a discrete latent representation, we incorporate ideas from vector quantisation (VQ). Using the VQ method allows the model to circumvent issues of "posterior collapse" -- where the latents are ignored when they are paired with a powerful autoregressive decoder -- typically observed in the VAE framework. Pairing these representations with an autoregressive prior, the model can generate high quality images, videos, and speech as well as doing high quality speaker conversion and unsupervised learning of phonemes, providing further evidence of the utility of the learnt representations. 


## Requirements
- Python 3
- [TensorFlow 2.0+](https://www.tensorflow.org/)
- [Matplotlib](https://matplotlib.org/)


## Installing the dependencies
Install virtualenv and creating a new virtual environment:

    pip install virtualenv
    virtualenv -p /usr/bin/python3 ./venv

Install dependencies

    pip3 install -r requirements.txt



## Disclaimer
This is not an official implementation.


## Citation
If you find this code useful for your research, please cite:

    @inproceedings{van2017neural,
      title={Neural discrete representation learning},
      author={van den Oord, Aaron and Vinyals, Oriol and others},
      booktitle={Advances in Neural Information Processing Systems},
      pages={6306--6315},
      year={2017}
    }
