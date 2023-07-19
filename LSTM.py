import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

# define model

class PropertyPrediction(nn.Module):

    def __init__(self, in_dim, h_dim, device):
        super(PropertyPrediction, self).__init__()
        self.device = device
        self.input_size  = in_dim
        self.hidden_size = h_dim
        self.output_size = in_dim

        self.lstm    = nn.LSTMCell(self.input_size, self.hidden_size)
        self.linear  = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, future=0, y=None):
        outputs = []
        b_size, seq_len, f_size = inputs.shape

        if y is not None:
            # how many steps to predict autoregressively:
            # future = 0 #random.randint(0, seq_len-1)
            future = random.randint(1, int(seq_len) // 2)
            # number of steps to predict from known conditions
            limit  = seq_len - future
        else:
            limit = seq_len

        # set the state of LSTM
        h_t = torch.zeros(inputs.size(0), self.hidden_size, dtype=torch.float32).to(self.device)
        c_t = torch.zeros(inputs.size(0), self.hidden_size, dtype=torch.float32).to(self.device)

        # predict using known conditions
        for i, input_t in enumerate(inputs[:, :limit].chunk(inputs.size(1), dim=1)):
            h_t, c_t   = self.lstm(torch.squeeze(input_t,1), (h_t, c_t))
            output     = self.linear(h_t)
            outputs   += [output]

        # predict autoregressively with teacher forcing
        for i in range(future):
            if y is not None and random.random() > 0.5:
                output = torch.squeeze(y[:,[limit+i],:], 1)
            h_t, c_t   = self.lstm(output, (h_t, c_t))
            output     = self.linear(h_t)
            outputs   += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

class gLV_NN_model(nn.Module):

    def __init__(self, in_dim, h_dim, out_dim, device):
        super(gLV_NN_model, self).__init__()
        self.device = device
        self.input_size  = in_dim
        self.hidden_size = h_dim
        self.output_size = out_dim

        self.linear_gLV  = nn.Linear(self.input_size, self.input_size, bias=True)
        self.linear_NN_h = nn.Linear(self.input_size, self.hidden_size)
        self.linear_NN_o = nn.Linear(self.hidden_size, self.output_size)

        # init gLV parameters to avoid exploiding species trajectories
        with torch.no_grad():
            for n, p in self.named_parameters():
                if 'gLV.weight' in n:
                    # matrix of interaction coefficients
                    p.copy_(torch.zeros(in_dim, in_dim).to(device))
                if 'gLV.bias' in n:
                    # vector of growth rates
                    p.copy_(torch.zeros(in_dim).to(device))

    def forward(self, inputs, y=None):
        # inputs should have shape [Batch Size, Time points, n_species+n_outputs]

        # gLV to estimate next step species abundance
        out = torch.relu(inputs[:,:,:self.input_size] + inputs[:,:,:self.input_size]*(self.linear_gLV(inputs[:,:,:self.input_size])))

        if self.output_size > 0:
            # NN to estimate metabolites
            met_out = self.linear_NN_o(torch.relu(self.linear_NN_h(out)))

            # concatenate outputs
            out = torch.cat((out, met_out), -1)

        return out

    def predict(self, inputs, future):
        # inputs should have shape [Batch Size, 1, n_species+n_outputs]
        # ^ only include initial condition

        # auto-regressive prediction of outputs
        outputs = []
        for i in range(future):
            outputs.append(self.forward(inputs))
            inputs = outputs[-1]

        return torch.cat(outputs, 1)

# 3. Create a PyTorch dataloader to handle creation of training batches

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, sys_vars, sys_scaler):

        # save the df with community data
        self.df = df
        self.sys_vars = sys_vars

        # set unique experiments
        self.comms = np.unique(df['Experiments'].values)

        # set scaler
        self.sys_scaler = sys_scaler

    def __getitem__(self, index):
        # get community
        community = self.comms[index]

        # pull community trajectory
        comm_inds = np.in1d(self.df['Experiments'].values, community)
        D = self.df.iloc[comm_inds].sort_values(by='Time', ascending=True)[self.sys_vars].values

        # standardize data
        X = self.sys_scaler.transform(D)

        # pull features data
        X = torch.tensor(X, dtype=torch.float32)

        # return inputs and output
        x  = X[:-1]
        y  = X[1:]
        return x, y

    def __len__(self):
        # total size of your dataset.
        return len(self.comms)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, X, exp_names, sys_scaler):
        # X is matrix with shape [N samples, N time points, Input dimension]
        self.X = X
        # set scaler on X
        self.sys_scaler = sys_scaler
        # set experiment id
        self.exp_names = exp_names

    def __getitem__(self, index):
        X_i = torch.tensor(self.sys_scaler.transform(self.X[index])[0], dtype=torch.float32)
        exp_name = self.exp_names[index]
        return X_i.unsqueeze(0), exp_name

    def __len__(self):
        # total size of your dataset.
        return self.X.shape[0]

### Organize data into pandas df

def format_data(df, sys_vars):

    # get experiment names
    experiments = df.Experiments.values

    # get unique experiments and number of time measurements
    unique_exps, counts = np.unique(experiments, return_counts=True)

    # determine time vector corresponding to longest sampled experiment
    exp_longest = unique_exps[np.argmax(counts)]
    exp_longest_inds = np.in1d(experiments, exp_longest)
    t_eval = df.iloc[exp_longest_inds]['Time'].values

    # initialize data matrix with NaNs
    D = np.empty([len(unique_exps), len(t_eval), len(sys_vars)])
    D[:] = np.nan

    # fill in data for each experiment
    for i,exp in enumerate(unique_exps):
        exp_inds  = np.in1d(experiments, exp)
        comm_data = df.copy()[exp_inds]

        # store data
        exp_time = comm_data['Time'].values
        sampling_inds = np.in1d(t_eval, exp_time)
        D[i][sampling_inds] = comm_data[sys_vars].values

    return D, unique_exps

def format_test_data(test_df, sys_vars):
    ### formats test data to only include initial condition ###

    # get experiment names
    experiments = test_df.Experiments.values

    # get unique experiments and number of time measurements
    unique_exps, seq_lengths = np.unique(experiments, return_counts=True)

    # determine unique sequence lengths
    unique_seq_lengths = np.unique(seq_lengths)

    # each batch has same sequence length
    X_batch = []
    seq_length_batch = []
    exp_name_batch = []
    for seq_length in unique_seq_lengths:
        # get all experiments with seq_length number of time points
        exps = unique_exps[seq_lengths==seq_length]
        X_set = np.zeros([len(exps), 1, len(sys_vars)])
        for i, exp in enumerate(exps):
            exp_df = test_df.iloc[experiments==exp].copy()
            exp_df.sort_values(by='Time', inplace=True)
            # pull only the initial condition
            X_set[i,0,:] = exp_df[sys_vars].values[0]
        X_batch.append(X_set)
        # record number of steps to predict forward
        seq_length_batch.append(seq_length-1)
        exp_name_batch.append(exps)

    return X_batch, seq_length_batch, exp_name_batch

# Define training function

def NNtrain(model, trainloader, optimizer, criterion, device, num_epochs):

    # training settings
    lr_decay       = 0.25
    decay_interval = 12
    # scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', verbose=True)

    for epoch in range(num_epochs):

        # decrease learning rate
        if (epoch+1) % decay_interval == 0:
            if epoch <= 99:
                optimizer.param_groups[0]['lr'] *= lr_decay

        # set up model for training
        train_loss = 0.
        # set training mode
        model.train()
        for i, (x, y) in enumerate(trainloader):

            # send data to gpu or cpu RAM
            # input has dimensions (batch size, n inputs)
            x, y = x.to(device), y.to(device)

            # Forward pass
            output = model(x, y=y)
            output = torch.reshape(output, y.shape)

            # zero gradients
            optimizer.zero_grad()

            # Compute loss
            loss = criterion(output, y)

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Keep track of training loss
            train_loss += loss.item()

        # # adjust learning rate
        # scheduler.step(train_loss)

        # print progress
        # if (epoch+1)%50==0:
        print("Epoch: {}/{}, Train Loss: {:.5f}".format(epoch+1, num_epochs, train_loss))

def gLVtrain(model, trainloader, optimizer, criterion, device, num_epochs):

    # training settings
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', verbose=True)

    for epoch in range(num_epochs):

        # set up model for training
        train_loss = 0.
        # set training mode
        model.train()
        for i, (x, y) in enumerate(trainloader):

            # send data to gpu or cpu RAM
            # input has dimensions (batch size, n inputs)
            x, y = x.to(device), y.to(device)

            # Forward pass
            output = model(x, y=y)
            output = torch.reshape(output, y.shape)

            # zero gradients
            optimizer.zero_grad()

            # Compute loss
            loss = criterion(output, y)

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Keep track of training loss
            train_loss += loss.item()

        # adjust learning rate
        scheduler.step(train_loss)

        # print progress
        # if (epoch+1)%50==0:
        print("Epoch: {}/{}, Train Loss: {:.5f}".format(epoch+1, num_epochs, train_loss))


class TimeSeriesMinMaxScaler():

    def __init__(self, minval=0., maxval=1.):
        self.minval = minval
        self.maxval = maxval
        self.range  = maxval-minval

    def fit(self, X):
        # X has dimensions: (N_experiments, N_timepoints, N_variables)
        self.X_min = np.zeros(X.shape[1:])
        self.X_max = np.nanmax(X, axis=0)
        self.X_range = self.X_max - self.X_min
        self.X_range[self.X_range==0.] = 1.
        return self

    def transform(self, X):
        # convert to 0-1 scale
        X_std = (X - self.X_min) / self.X_range
        # scale to set min-max scale
        X_scaled = X_std*self.range + self.minval
        return X_scaled

    def inverse_transform(self, X_scaled):
        X_std = (X_scaled - self.minval)/self.range
        X = X_std*self.X_range + self.X_min
        return X

    def inverse_transform_stdv(self, X_scaled):
        X_std = (X_scaled - self.minval)/self.range
        X = X_std*self.X_range
        return X

class TimeSeriesStandardScaler():

    def __init__(self, mean=0., std=1.):
        self.mean=mean
        self.std=std

    def fit(self, X):
        # X has dimensions: (N_experiments, N_timepoints, N_variables)
        self.mean = np.nanmean(X, axis=0)
        self.std  = 4*np.nanstd(X, axis=0)
        # center unchanging inputs to zero.
        self.std[self.std==0.] = 1.
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return X*self.std + self.mean

class IdentityScaler():
    def __init__(self):
        pass

    def fit(self, X):
        # X has dimensions: (N_experiments, N_timepoints, N_variables)
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X

### Define RNN CLASS here ###

class LSTM():
    def __init__(self, df, sys_vars, device = 'cuda',
                 hidden_size=4096, batch_size=10, lr=5e-3, iteration=200):
        '''
        df is a dataframe with columns
        ['Experiments', 'Time', 'S_1', ..., 'S_M']

        hidden_size := size of neural network hidden layer
        batch_size  := number of samples to include in each training batch
        '''

        # set device
        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')

        # nf := number of features
        self.nf = len(sys_vars)
        self.sys_vars = np.array(sys_vars)
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lr = lr
        self.iteration = iteration
        self.comms = np.unique(df['Experiments'].values)

        # format data and initialize data scalers
        X, exp_names = format_data(df, self.sys_vars)
        # fit scalers to training data
        self.sys_scaler = TimeSeriesStandardScaler().fit(X)
        #self.sys_scaler = TimeSeriesMinMaxScaler().fit(X)

    def train(self, train_df):

        # initialize the data set loaders
        traindataset = Dataset(train_df, self.sys_vars, self.sys_scaler)
        trainloader = torch.utils.data.DataLoader(dataset=traindataset,
                                                  batch_size=self.batch_size)

        # initialize model
        self.rnn = PropertyPrediction(len(self.sys_vars), self.hidden_size, self.device).to(self.device)
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = torch.nn.MSELoss()

        ### train nn
        NNtrain(self.rnn, trainloader, optimizer, criterion,
                device=self.device, num_epochs=self.iteration)

    def predict(self, test_df):
        # init dataframe to return
        return_df = pd.DataFrame()

        # format test data
        X_batch, seq_length_batch, exp_name_batch = format_test_data(test_df, self.sys_vars)

        # initialize the data set loaders
        with torch.no_grad():
            for X, seq_length, exp_names in zip(X_batch, seq_length_batch, exp_name_batch):
                testdataset = TestDataset(X, exp_names, self.sys_scaler)
                testloader  = torch.utils.data.DataLoader(dataset=testdataset, batch_size=self.batch_size)
                for X_i, exps in testloader:
                    # send to device
                    X_i = X_i.to(self.device)
                    # make future predictions
                    X_pred = self.rnn(X_i, future=seq_length-1)
                    # set initial value
                    X_pred = torch.cat((X_i, X_pred), dim=1)
                    # inverse scale
                    X_pred = self.sys_scaler.inverse_transform(X_pred.cpu())
                    # save to dataframe
                    for i, exp in enumerate(exps):
                        test_df_exp = test_df.iloc[test_df.Experiments.values==exp].copy()
                        test_times = test_df_exp['Time'].values
                        exp_df = pd.DataFrame()
                        exp_df['Experiments'] = [exp]*len(test_times)
                        exp_df['Time'] = test_times
                        for j, feature in enumerate(self.sys_vars):
                            exp_df[feature] = test_df_exp[feature].values
                            exp_df[feature + ' pred'] = np.clip(X_pred[i,:,j], 0., np.inf)
                        return_df = pd.concat((return_df, exp_df))
        return return_df

class gLV_NN():
    def __init__(self, df, species, metabolites, device = 'cuda',
                 hidden_size=4096, batch_size=10, lr=5e-3, iteration=200):
        '''
        df is a dataframe with columns
        ['Experiments', 'Time', 'S_1', ..., 'S_M']

        hidden_size := size of neural network hidden layer
        batch_size  := number of samples to include in each training batch
        '''

        # set device
        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')

        # nf := number of features
        self.species = species
        self.metabolites = metabolites
        self.sys_vars = np.concatenate((np.array(species), np.array(metabolites)))
        self.nf = len(self.sys_vars)
        if len(metabolites) > 0:
            self.hidden_size = hidden_size
        else:
            # no need for NN if not predicting metabolites
            self.hidden_size = 1
        self.batch_size = batch_size
        self.lr = lr
        self.iteration = iteration
        self.comms = np.unique(df['Experiments'].values)

        # format data and initialize data scalers
        X, exp_names = format_data(df, self.sys_vars)
        # fit scalers to training data
        # self.sys_scaler = TimeSeriesMinMaxScaler().fit(X)
        self.sys_scaler = IdentityScaler().fit(X)

    def train(self, train_df):

        # initialize the data set loaders
        traindataset = Dataset(train_df, self.sys_vars, self.sys_scaler)
        trainloader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=self.batch_size)

        # initialize model
        self.rnn = gLV_NN_model(len(self.species), self.hidden_size, len(self.metabolites), self.device).to(self.device)
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss()

        ### train nn
        gLVtrain(self.rnn, trainloader, optimizer, criterion,
                device=self.device, num_epochs=self.iteration)

    def predict(self, test_df):
        # init dataframe to return
        return_df = pd.DataFrame()

        # format test data
        X_batch, seq_length_batch, exp_name_batch = format_test_data(test_df, self.sys_vars)

        # initialize the data set loaders
        with torch.no_grad():
            for X, seq_length, exp_names in zip(X_batch, seq_length_batch, exp_name_batch):
                testdataset = TestDataset(X, exp_names, self.sys_scaler)
                testloader  = torch.utils.data.DataLoader(dataset=testdataset, batch_size=self.batch_size)
                for X_i, exps in testloader:
                    # send to device
                    X_i = X_i.to(self.device)
                    # make future predictions
                    X_pred = self.rnn.predict(X_i, future=seq_length)
                    # set initial value
                    X_pred = torch.cat((X_i, X_pred), dim=1)
                    # inverse scale
                    X_pred = self.sys_scaler.inverse_transform(X_pred.cpu())
                    # save to dataframe
                    for i, exp in enumerate(exps):
                        test_df_exp = test_df.iloc[test_df.Experiments.values==exp].copy()
                        test_times = test_df_exp['Time'].values
                        exp_df = pd.DataFrame()
                        exp_df['Experiments'] = [exp]*len(test_times)
                        exp_df['Time'] = test_times
                        for j, feature in enumerate(self.sys_vars):
                            exp_df[feature] = test_df_exp[feature].values
                            exp_df[feature + ' pred'] = np.clip(X_pred[i,:,j], 0., np.inf)
                        return_df = pd.concat((return_df, exp_df))
        return return_df
