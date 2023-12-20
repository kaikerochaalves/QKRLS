# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:27:20 2021

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""
# Importing libraries
import pandas as pd
import numpy as np

class QKRLS:
    def __init__(self, epsilon = 0.1, sigma = 1, gamma = 0.01):
        # Parameters: K is the Gram matrix K, alpha is the vector of parameters, 
        # P is the inverse of K (equivalent to Kinv or Q in other algorithms), 
        # m is the number of elements in the dictionary, C is the dictionary
        # and UppercaseLambda is a vector
        self.parameters = pd.DataFrame(columns = ['K', 'alpha', 'P', 'm', 'C', 'UppercaseLambda'])
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
        # Computing the residual square in the testing phase
        self.ResidualTestPhase = np.array([])
        # Hyperparameters and parameters
        self.sigma = sigma
        # Quantization size
        self.epsilon = epsilon
        # Regularization parameter
        self.gamma = gamma
         
    def fit(self, X, y):

        # Compute the number of samples
        n = X.shape[0]
        
        # Initialize the first input-output pair
        x0 = X[0,].reshape(-1,1)
        y0 = y[0]
        
        # Initialize QKRLS
        self.Initialize_QKRLS(x0, y0)

        for k in range(1, n):

            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
                      
            # Update QKRLS
            h = self.QKRLS(x, y[k])
            
            # Compute output
            Output = self.parameters.loc[0, 'alpha'].T @ h
            
            # Store results
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output )
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(y[k]) - Output )
        return self.OutputTrainingPhase
            
    def predict(self, X):

        for k in range(X.shape[0]):
            
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T

            # Compute k
            h = np.array(())
            for ni in range(self.parameters.loc[0, 'C'].shape[1]):
                h = np.append(h, [self.Kernel(self.parameters.loc[0, 'C'][:,ni].reshape(-1,1), x)])
            h = h.reshape(h.shape[0],1)
            
            # Compute the output
            Output = self.parameters.loc[0, 'alpha'].T @ h
            
            # Store the output
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output )
            
        return self.OutputTestPhase

    def Kernel(self, x1, x2):
        k = np.exp( - ( 1/2 ) * ( (np.linalg.norm( x1 - x2 ))**2 ) / ( self.sigma**2 ) )
        return k
    
    def Initialize_QKRLS(self, x, y):
        k11 = self.Kernel(x, x)
        K = np.ones((1,1)) * ( k11 + self.gamma )
        P = np.ones((1,1)) / ( k11 + self.gamma )
        alpha = np.ones((1,1)) * y / k11
        UppercaseLambda = np.ones((1,1))
        NewRow = pd.DataFrame([[K, alpha, P, 1., x, UppercaseLambda]], columns = ['K', 'alpha', 'P', 'm', 'C', 'UppercaseLambda'])
        self.parameters = pd.concat([self.parameters, NewRow], ignore_index=True)
        # Initialize first output and residual
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, y)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, 0.)
        
    def QKRLS(self, x, y):
        i = 0
        # Compute h (k)
        h = np.array(())
        for ni in range(self.parameters.loc[i, 'C'].shape[1]):
            h = np.append(h, [self.Kernel(self.parameters.loc[i, 'C'][:,ni].reshape(-1,1), x)])
        ht = h.reshape(-1,1)
        htt = self.Kernel(x, x)
        # Searching for the lowest distance between the input and the dictionary inputs
        distance = []
        for ni in range(self.parameters.loc[i, 'C'].shape[1]):
            distance.append(np.linalg.norm(self.parameters.loc[i, 'C'][:,ni].reshape(-1,1) - x))
        # Find the index of minimum distance
        j = np.argmin(distance)
        # Novelty criterion
        if distance[j] <= self.epsilon:
            # Update Uppercase Lambda
            xi = np.zeros(self.parameters.at[i, 'alpha'].shape)
            xi[j] = 1.
            self.parameters.at[i, 'UppercaseLambda'] = self.parameters.loc[i, 'UppercaseLambda'] + xi @ xi.T
            # Compute Pj and Kj
            Pj = self.parameters.loc[i, 'P'][:,j].reshape(-1,1)
            Kj = self.parameters.loc[i, 'K'][:,j].reshape(-1,1)
            # Update P - P is the inverse of K (equivalent to Kinv or Q in other algorithms)
            self.parameters.at[i, 'P'] = self.parameters.loc[i, 'P'] - ( Pj @ (Kj.T @ self.parameters.loc[i, 'P'] ) ) / ( 1 + Kj.T @ Pj )
            # Updating alpha
            self.parameters.at[i, 'alpha'] = self.parameters.loc[i, 'alpha'] + Pj @ ( y - Kj.T @ self.parameters.loc[i, 'alpha'] ) / ( ( 1 + Kj.T @ Pj ) )
        else:
            # Update the dictionary
            self.parameters.at[i, 'C'] = np.hstack([self.parameters.loc[i,  'C'], x])
            # Compute z, z_upperlambda, and r
            z = self.parameters.loc[i, 'P'] @ ht
            z_upperlambda = self.parameters.loc[i, 'P'] @ self.parameters.loc[i, 'UppercaseLambda'] @ ht
            r = self.gamma + htt - ht.T @ z_upperlambda
            # Compute the estimated error
            e = y - ht.T @ self.parameters.loc[i, 'alpha']
            self.parameters.at[i, 'm'] = self.parameters.loc[i, 'm'] + 1
            # Update K                      
            self.parameters.at[i, 'K'] = np.lib.pad(self.parameters.loc[i,  'K'], ((0,1),(0,1)), 'constant', constant_values=(0))
            sizeK = self.parameters.loc[i,  'K'].shape[0] - 1
            self.parameters.at[i, 'K'][sizeK,sizeK] = htt
            self.parameters.at[i, 'K'][0:sizeK,sizeK] = ht.flatten()
            self.parameters.at[i, 'K'][sizeK,0:sizeK] = ht.flatten()
            # Update UppercaseLambda
            self.parameters.at[i, 'UppercaseLambda'] = np.lib.pad(self.parameters.loc[i, 'UppercaseLambda'], ((0,1),(0,1)), 'constant', constant_values=(0))
            sizeUppercaseLambda = self.parameters.loc[i,  'UppercaseLambda'].shape[0] - 1
            self.parameters.at[i, 'UppercaseLambda'][sizeUppercaseLambda,sizeUppercaseLambda] = 1.
            # Update P - P is the inverse of K (equivalent to Kinv or Q in other algorithms)
            self.parameters.at[i, 'P'] = self.parameters.loc[i,  'P'] * r + z_upperlambda @ z.T
            self.parameters.at[i, 'P'] = np.lib.pad(self.parameters.loc[i, 'P'], ((0,1),(0,1)), 'constant', constant_values=(0))
            sizeP = self.parameters.loc[i, 'P'].shape[0] - 1
            self.parameters.at[i, 'P'][sizeP,sizeP] = htt
            self.parameters.at[i, 'P'][0:sizeP,sizeP] = - z_upperlambda.flatten()
            self.parameters.at[i, 'P'][sizeP,0:sizeP] = - z_upperlambda.flatten()
            self.parameters.at[i, 'P'] = ( 1 / r ) * self.parameters.loc[i, 'P']
            # Update alpha
            self.parameters.at[i, 'alpha'] = self.parameters.loc[i, 'alpha'] - ( z_upperlambda * ( 1 / r ) * e )
            self.parameters.at[i, 'alpha'] = np.vstack([self.parameters.loc[i, 'alpha'], ( 1 / r ) * e ])
            h = np.append(ht, htt)
            
        return h