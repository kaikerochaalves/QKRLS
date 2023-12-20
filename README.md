# QKRLS (quantized kernel recursive least squares)

The quantized kernel recursive least squares (QKRLS) is a model proposed by Chen et al. [1].

- [QKRLS](https://github.com/kaikerochaalves/QKRLS/blob/6a1dcf72ecebd8473dc447df32b17ebb5b91e67a/Model/QKRLS.py) is the QKRLS model.

- [GridSearch_AllDatasets](https://github.com/kaikerochaalves/QKRLS/blob/6a1dcf72ecebd8473dc447df32b17ebb5b91e67a/GridSearch_AllDatasets.py) is the file to perform a grid search for all datasets and store the best hyper-parameters.

- [Runtime_AllDatasets](https://github.com/kaikerochaalves/QKRLS/blob/6a1dcf72ecebd8473dc447df32b17ebb5b91e67a/Runtime_AllDatasets.py) perform 30 simulations for each dataset and compute the mean runtime and the standard deviation.

- [MackeyGlass](https://github.com/kaikerochaalves/QKRLS/blob/6a1dcf72ecebd8473dc447df32b17ebb5b91e67a/MackeyGlass.py) is the script to prepare the Mackey-Glass time series, perform simulations, compute the results and plot the graphics. 

- [Nonlinear](https://github.com/kaikerochaalves/QKRLS/blob/6a1dcf72ecebd8473dc447df32b17ebb5b91e67a/Nonlinear.py) is the script to prepare the nonlinear dynamic system identification time series, perform simulations, compute the results and plot the graphics.

- [LorenzAttractor](https://github.com/kaikerochaalves/QKRLS/blob/6a1dcf72ecebd8473dc447df32b17ebb5b91e67a/LorenzAttractor.py) is the script to prepare the Lorenz Attractor time series, perform simulations, compute the results and plot the graphics. 

[1] B. Chen, S. Zhao, P. Zhu, J. C. Principe, Quantized kernel recursive least squares algorithm, IEEE Transactions on Neural Networks and Learning Systems 24 (9) (2013) 1484–1491. doi:https://doi.org/10.1109/TNNLS.952013.2258936
