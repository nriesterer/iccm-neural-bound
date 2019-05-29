import time

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import ccobra

import onehot

def to_onehot(df):
    train_x = []
    train_y = []

    for subj_id, subj_df in df.groupby('id'):
        subj_x = []
        subj_y = []

        for _, task_series in subj_df.sort_values('sequence').iterrows():
            item = ccobra.Item(
                task_series['id'],
                task_series['domain'],
                task_series['task'],
                task_series['response_type'],
                task_series['choices'],
                task_series['sequence'])
            syllogism = ccobra.syllogistic.Syllogism(item)

            # Encode the task input
            task = onehot.onehot_syllogism_content(syllogism.encoded_task)

            # Encode the response output
            encoded_response = syllogism.encode_response(task_series['response'].split(';'))
            resp = onehot.onehot_response(encoded_response)

            subj_x.append(task)
            subj_y.append(resp)

        train_x.append(subj_x)
        train_y.append(subj_y)

    train_x = torch.from_numpy(np.array(train_x)).float()
    train_y = torch.from_numpy(np.array(train_y)).float()

    return train_x, train_y

# Configuration
train_file = '../../data/Ragni-train.csv'
test_file = '../../data/Ragni-test.csv'
n_epochs = 150

# Load the data
train_df = pd.read_csv(train_file)
train_x, train_y = to_onehot(train_df)

test_df = pd.read_csv(test_file)
test_x, test_y = to_onehot(test_df)

# Train the model
class RNN(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, output_size=9):
        super(RNN, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2)
        self.h2o = nn.Linear(hidden_size, 9)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.h2o(output)
        return output, hidden

net = RNN()
optimizer = optim.Adam(net.parameters())
criterion = nn.CrossEntropyLoss()

# Train the network
train_accs = []
test_accs = []
losses = []
for epoch in range(n_epochs):
    start_time = time.time()
    net.train()

    # Shuffle the training data
    perm_idxs = np.random.permutation(np.arange(len(train_x)))
    train_x = train_x[perm_idxs]
    train_y = train_y[perm_idxs]

    # Loop over the training instances
    epoch_losses = []
    for idx in range(len(train_x)):
        cur_x = train_x[idx]
        cur_y = train_y[idx]

        input = cur_x.view(64, 1, -1)
        outputs, _ = net(input, None)

        # Backpropagation and parameter optimization
        loss = criterion(outputs.view(64, -1), cur_y.argmax(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    losses.append(np.mean(epoch_losses))

    # Print statistics
    print('Epoch {}/{} ({:.2f}s): {:.4f} ({:.4f})'.format(
        epoch + 1, n_epochs, time.time() - start_time, np.mean(epoch_losses), np.std(epoch_losses)))

    # Test the predictive accuracy
    net.eval()
    epoch_train_accs = []
    for subj_idx in range(len(train_x)):
        pred, _ = net(train_x[subj_idx].view(64, 1, -1), None)
        pred_max = pred.view(64, -1).argmax(1)
        truth = train_y[subj_idx].argmax(1)

        acc = torch.mean((pred_max == truth).float()).item()
        epoch_train_accs.append(acc)

    print('   train: {:.4f} ({:.4f})'.format(np.mean(epoch_train_accs), np.std(epoch_train_accs)))
    train_accs.append(epoch_train_accs)

    epoch_test_accs = []
    for subj_idx in range(len(test_x)):
        pred, _ = net(test_x[subj_idx].view(64, 1, -1), None)
        pred_max = pred.view(64, -1).argmax(1)
        truth = test_y[subj_idx].argmax(1)

        acc = torch.mean((pred_max == truth).float()).item()
        epoch_test_accs.append(acc)

    print('   test acc: {:.4f} ({:.4f})'.format(np.mean(epoch_test_accs), np.std(epoch_test_accs)))
    test_accs.append(epoch_test_accs)

# Store the accuracies
np.save('train_accs.npy', np.array(train_accs))
np.save('test_accs.npy', np.array(test_accs))
np.save('train_losses.npy', np.array(losses))
