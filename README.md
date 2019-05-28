# Outlier Detection for Time Series with Recurrent Autoencoder Ensembles
This is a TensorFlow implementation of Outlier Detection for Time Series with Recurrent Autoencoder Ensembles in the following paper: 
Tung Kieu, Bin Yang, Chenjuan Guo, Christian S. Jensen, _Outlier Detection for Time Series with Recurrent Autoencoder Ensembles_, __IJCAI 2019__.

# Requirements
* Python 3.x
* Numpy
* Pandas
* TensorFlow
* Scikit-learn

# Dataset
We use two dataset __NAB__ and __ECG__ that is a public dataset. You can follow the links in the paper to download the dataset.

# Model
We propose two model __IF__ and __SF__

## IF
<img src="https://github.com/tungsomot/OED/blob/master/S-RNN-AE.png" width="600">

## SF
<img src="https://github.com/tungsomot/OED/blob/master/SC-S-RNN-AE.png" width="600">

# Citation
If you find this repository, e.g., the code and the datasets, useful in your research, please cite the following paper:

```
@inproceedings{tungbcc19,
  title={Outlier Detection for Time Series with Recurrent Autoencoder Ensembles},
  author={Kieu, Tung and Yang, Bin and Guo, Chenjuan and S. Jensen, Christian},
  booktitle={International Joint Conference on Artificial Intelligence (IJCAI '19)},
  year={2019}
}
``` 
