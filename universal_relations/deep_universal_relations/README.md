# deep-universal-relations


Relations between stellar properties independent of the nuclear equation of state offer profound insights into neutron star physics and have practical applications in data analysis by breaking the degeneracy between the parameters of interest. Commonly, these relations are derived from utilizing various realistic nuclear cold hadronic, hyperonic, and hybrid EoS models, each of which should obey the current multimessenger constraints and cover a wide range of stiffnesses. Concurrently, the field of multimessenger astronomy has been significantly enhanced by the advent of gravitational wave astronomy, which increasingly incorporates deep learning techniques and algorithms. At the same time, X-ray spectral data from NICER based on known pulsars are available, and additional observations are expected from upcoming missions. Therefore, in the coming years, a wealth of information on neutron stars is expected to be gathered from multiple observational channels. In this study, we revisit established universal relations, introduce new ones, and reassess them using a feed-forward neural network as a regression model. More specifically, we mainly propose "deep" EoS-insensitive hypersurface relations for rapidly rotating compact objects between several of the star's global parameters, such as the moment of inertia, the reciprocal of the moment of inertia, the rotational frequency, mass and current multipole moments and the equatorial radius, which achieve an accuracy of within 1 % in most cases, with only a small fraction of investigated models exceeding this threshold. While analytical expressions can be used to represent some of these relations, the neural network approach demonstrates superior performance, particularly in complex regions of the parameter space. Furthermore, we use the SHapley Additive exPlanations (SHAP) method to interpret the suggested feed-forward network's predictions, since SHAP is based on a strong theoretical framework inspired by the field of cooperative Game Theory. Most importantly, these highly accurate universal relations empowered with the interpretability description could be used in efforts to constrain the high-density equation of state in neutron stars, with the potential to enhance our understanding as new observables emerge.

### Summary of the EoS-insensitive relations investigated in this work

| Relation | Suggested ANN | Max % Error |
|:--------:|:---------------------------:|:-------------:|
| $\bar{I}(\chi,\bar{Q})$ | Improved model | $2.76\%$ |
| $\mathcal{D}(\chi,\log\bar{Q})$ | Improved model | $2.70\%$ |
| $\mathcal{D}(C,\chi,\log\bar{Q})$ |  New model | $2.35\%$ | 
| $(M\times\hat{f})(\chi, C)$ | New model | $1.77\%$ | 
| $\bar{Q}(C,\chi,\mathcal{D})$ | New model | $2.32\%$ |
| $\bar{S}_3(\log\bar{Q})$ |  model | $6.64\%$ | 
| $\bar{S}_3(\chi,\mathcal{D},\log\bar{Q})$ | New model | $3.32\%$ | 
| $\bar{M}_4(\log\bar{Q})$ | Improved model | $14.34\%$ | 
| $\left(\bar{M}_4/\bar{S}_3\right)(\log\bar{Q})$ | Improved model | $13.22\%$ |
| $\left(\bar{M}_4/\bar{S}_3\right)(\chi, C,\bar{Q})$ | New model | $9.19\%$ | 
| $\bar{M}_4(\chi, C,\bar{Q},\bar{S}_3$) | New model | $8.11\%$ |
| $\bar{M}_4(C,\bar{Q},\bar{S}_3$) | New model | $7.52\%$ | 
| $\bar{\mathcal{R}}(\bar{M},\chi,\bar{Q},\bar{S}_3)$ | New model | $3.81\%$ |


### Repository Overview
This repository contains the ANN model architecture for each case, pre-trained model weights, and example notebooks that demonstrate how to use the suggested regression models for predicting the neutron star global properties discussed in the paper. In each case, a small subset of the test dataset is used.

### Contents
* **Model Architecture**: The machine learning models developed for regression tasks.
* **Pre-trained Weights**: Ready-to-use model weights for reproducing our results.
* **Demos**: Sample scripts illustrating how to load the models and perform predictions on the data.

### Installation
To set up the necessary environment for running the code, ensure the following dependencies are installed:

```bash
pip install numpy pandas scikit-learn torch
```

### Dependencies
* `numpy`
* `pandas`
* `sklearn`
* `pytorch`

### Usage
1. **Loading the Pre-trained Model**: You can load the provided pre-trained model weights and use them to predict surface parameters of neutron stars. Check the demos folder for example usage.
```python
import torch
model = torch.load('path_to_model_weights.pth')
```
2. **Running a Demo**:These scripts demonstrate how estimate the neutron star properties related to the star's surface using universal relations.

### Contact
For any questions, feel free to contact us via email at gpapigki@auth.gr, g.vardakas@uoi.gr

### Citation
If you find this work useful in your research, please cite our paper as:
```
@article{papigkiotis2025assessing,
      title={Assessing Universal Relations for Rapidly Rotating Neutron Stars: Insights from an Interpretable Deep Learning Perspective}, 
      author={Grigorios Papigkiotis and Georgios Vardakas and Nikolaos Stergioulas},
      year={2025},
      eprint={2508.05850},
      archivePrefix={arXiv},
      primaryClass={astro-ph.HE},
      url={https://arxiv.org/abs/2508.05850}, 
}
```


