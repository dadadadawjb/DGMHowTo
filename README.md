# DGMHowTo
A collection of vanilla **Deep Generative Models** (**DGM**) re-implementation with clean and well-annotated PyTorch implementation for systematic learning toward Deep Generative Models.

## Support Models
* **[Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)** (**GAN**)
* **[Variational Auto-Encoder](https://arxiv.org/abs/1312.6114)** (**VAE**)

## Get Started
```bash
# install dependencies
pip install -r requirements.txt

# prepare your dataset in `data`

# prepare your experiment configuration in `configs`

# train the model
python train_*.py --config configs/*.txt
```

## Results
### GAN
* loss process graph
![gan_loss](assets/gan_train_loss.png)
* noise generation process
![gan_generation](assets/gan_train_eval_generation.gif)
* noise discriminator accuracy process
![gan_accuracy](assets/gan_train_eval_acc.png)

### VAE
* loss process graph
![vae_loss](assets/vae_train_loss.png)
* noise generation process
![vae_generation](assets/vae_train_eval_generation.gif)

## Note
Kudos to the authors for their amazing results.
```bib
@misc{https://doi.org/10.48550/arxiv.1312.6114,
  doi = {10.48550/ARXIV.1312.6114},
  title = {Auto-Encoding Variational Bayes},
  author = {Kingma, Diederik P and Welling, Max},
  year = {2013},
  publisher = {arXiv},
}
```
```bib
@misc{https://doi.org/10.48550/arxiv.1406.2661,
  doi = {10.48550/ARXIV.1406.2661},
  title = {Generative Adversarial Networks},
  author = {Goodfellow, Ian J. and Pouget-Abadie, Jean and Mirza, Mehdi and Xu, Bing and Warde-Farley, David and Ozair, Sherjil and Courville, Aaron and Bengio, Yoshua},
  year = {2014},
  publisher = {arXiv},
}
```
