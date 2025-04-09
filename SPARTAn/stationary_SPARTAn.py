import torch
import torch.nn as nn
import torch.optim as optim
def Stationary_Spartan(df,forecasting):
    Lambda = nn.Parameter(torch.zeros(1, df.shape[1]))#1, se è multivariata dovrei aumentare, forse dovrei mettere forecasting
    
