# mlcs-ex6-ann-speed-prediction

Prediction of pedestrian velocities using ANNs and Random Forest. Based on the paper [Prediction of Pedestrian Speed with Artificial Neural Networks](https://arxiv.org/abs/1801.09782) by Tordeux et. al.

The data was obtained from pedestrian experiments of the [Civil Safety Research Unit of Forschungszentrum Jülich](https://ped.fz-juelich.de/database/doku.php) and provided by Antoine Tordeux on [Zenodo.org](https://zenodo.org/record/1054017).

## Outline

Below serves as a quick overview about the content of the repository.

### `task_2.ipynb`
Loading the raw bottleneck dataset, visualizing it, and adding additional features:
- relative distances to k closest neighbors at a time during the experiment = `feautures`
- pedestrian velocity = `target`

### `task_3.ipynb`
Training an artifical neural network (ANN) with varying number and sizes of hidden layers to predict the velocity of each pedestrian at a time, by using the relative distance - based features as input.

### `task_4.ipynb`
Using a classical Random Forest Regressor to also predict the pedestrian velocity based on the relative distances. Serves as a comparison to the ANN model and achieved results.

