""" Evaluates the training performance of the MLP.

"""

import time
import copy

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import ccobra

import sylmlp as sa
import onehot

# Settings
training_datafile = '../../data/Ragni-train.csv'
test_datafile = '../../data/Ragni-test.csv'
n_epochs = 400
batch_size = 16
net = sa.SylMLP()
optimizer = optim.Adam(net.parameters())
criterion = nn.MSELoss()

def csv_to_tensor(datafile):
    inputs = []
    targets = []

    df = pd.read_csv(datafile)
    for _, subj_df in df.groupby('id'):
        subj_inputs = []
        subj_targets = []

        for _, task_series in subj_df.sort_values('sequence').iterrows():
            item = ccobra.Item(
                task_series['id'], task_series['domain'], task_series['task'],
                task_series['response_type'], task_series['choices'], task_series['sequence'])
            syllogism = ccobra.syllogistic.Syllogism(item)

            # Convert to onehot
            subj_inputs.append(onehot.onehot_syllogism_content(syllogism.encoded_task))
            subj_targets.append(onehot.onehot_response(
                syllogism.encode_response(task_series['response'].split(';'))))

        inputs.append(subj_inputs)
        targets.append(subj_targets)

    return torch.tensor(inputs).float(), torch.tensor(targets).float()

# Import the datasets
training_inputs, training_targets = csv_to_tensor(training_datafile)
test_inputs, test_targets = csv_to_tensor(test_datafile)

def evaluate_model(net, inputs, targets, adapt):
    accs = []
    for subj_idx, subj_data in enumerate(inputs):
        subj_net = copy.deepcopy(net)
        subj_optimizer = optim.Adam(subj_net.parameters())

        subj_accs = []

        for task_idx, task_data in enumerate(subj_data):
            # Update the training accuracy list
            output = subj_net(task_data)
            truth = targets[subj_idx, task_idx]
            subj_accs.append(output.argmax() == truth.argmax())

            # Perform the adaption
            if adapt:
                loss = criterion(output, truth)
                subj_optimizer.zero_grad()
                loss.backward()
                subj_optimizer.step()

        accs.append(subj_accs)
    return np.array(accs)

# Training loop
train_accs = []
train_accs_adapt = []
test_accs = []
test_accs_adapt = []
losses = []
for epoch in range(n_epochs):
    start_time = time.time()

    # Shuffle the training data
    rnd_idxs = np.random.permutation(np.arange(len(training_inputs)))
    training_inputs = training_inputs[rnd_idxs]
    training_targets = training_targets[rnd_idxs]

    # Convert to training dimensionality
    train_x = training_inputs.view(-1, 12)
    train_y = training_targets.view(-1, 9)

    batch_losses = []
    for batch_idx in range(len(train_x) // batch_size):
        start = batch_idx * batch_size
        end = start + batch_size

        batch_x = train_x[start:end]
        batch_y = train_y[start:end]

        outputs = net(batch_x)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())

    losses.append(np.mean(batch_losses))

    # Training dataset evaluation
    train_accs.append(evaluate_model(net, training_inputs, training_targets, False))
    train_accs_adapt.append(evaluate_model(net, training_inputs, training_targets, True))
    test_accs.append(evaluate_model(net, test_inputs, test_targets, False))
    test_accs_adapt.append(evaluate_model(net, test_inputs, test_targets, True))

    print('Epoch {}/{} ({:.2f}s)'.format(epoch + 1, n_epochs, time.time() - start_time))
    print('   train   : {:.4f} ({:.4f})'.format(np.mean(train_accs[-1]), np.std(train_accs[-1])))
    print('   train ad: {:.4f} ({:.4f})'.format(np.mean(train_accs_adapt[-1]), np.std(train_accs_adapt[-1])))
    print('   test    : {:.4f} ({:.4f})'.format(np.mean(test_accs[-1]), np.std(test_accs[-1])))
    print('   test  ad: {:.4f} ({:.4f})'.format(np.mean(test_accs_adapt[-1]), np.std(test_accs_adapt[-1])))
    print()

# Store the results
np.save('train_accs.npy', np.array(train_accs))
np.save('train_accs_adapt.npy', np.array(train_accs_adapt))
np.save('test_accs.npy', np.array(test_accs))
np.save('test_accs_adapt.npy', np.array(test_accs_adapt))
np.save('train_losses.npy', np.array(losses))
