""" Evaluates the training performance of the autoencoder.

"""

import time

import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

import ccobra

import onehot
import autoencoder


# General settings
training_datafile = '../../data/Ragni-train.csv'
test_datafile = '../../data/Ragni-test.csv'
n_epochs = 150
batch_size = 16
net = autoencoder.DenoisingAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

def csv_to_tensor(datafile):
    profiles = []
    response_dicts = []
    task_sequences = []

    df = pd.read_csv(datafile)
    for _, subj_df in df.groupby('id'):
        # Obtain the task-response mapping for all syllogisms
        response_dict = {}
        task_sequence = []
        for _, task_series in subj_df.sort_values('sequence').iterrows():
            item = ccobra.Item(
                task_series['id'], task_series['domain'], task_series['task'],
                task_series['response_type'], task_series['choices'], task_series['sequence'])
            syllogism = ccobra.syllogistic.Syllogism(item)

            response_dict[syllogism.encoded_task] = syllogism.encode_response(
                task_series['response'].split(';'))
            task_sequence.append(syllogism.encoded_task)

        # Convert the task-response mapping to the reasoner profile
        profile = []
        for task in ccobra.syllogistic.SYLLOGISMS:
            profile.append(onehot.onehot_response(response_dict[task]))
        profiles.append(profile)

        response_dicts.append(response_dict)
        task_sequences.append(task_sequence)

    profile_tensor = torch.tensor(profiles).float().view(-1, 576)
    return profile_tensor, np.array(response_dicts), np.array(task_sequences)

# Construct the training and test tensors
train_data, train_resp_dicts, train_seqs = csv_to_tensor(training_datafile)
test_data, test_resp_dicts, test_seqs = csv_to_tensor(test_datafile)

def compute_accuracy(data, resp_dicts, seqs):
    accs = []
    for subj_idx in range(len(data)):
        subj_resp_dict = resp_dicts[subj_idx]
        subj_seq = seqs[subj_idx]

        profile_tensor = torch.zeros((576)).float()

        subj_hits = []
        for task in subj_seq:
            task_idx = ccobra.syllogistic.SYLLOGISMS.index(task)
            start = task_idx * 9
            end = start + 9
            truth = subj_resp_dict[task]

            # Query the network for a prediction
            prediction_idx = net(profile_tensor)[start:end].argmax()
            prediction = ccobra.syllogistic.RESPONSES[prediction_idx]
            subj_hits.append(prediction == truth)

            # Add the true response to the profile
            profile_tensor[start:end] = torch.from_numpy(onehot.onehot_response(truth))

        accs.append(subj_hits)
    return accs

# Training loop
train_accs = []
test_accs = []
losses = []
for epoch in range(n_epochs):
    start_time = time.time()

    # Permute the training data
    rnd_idxs = np.random.permutation(np.arange(len(train_data)))
    train_data = train_data[rnd_idxs]
    train_resp_dicts = train_resp_dicts[rnd_idxs]
    train_seqs = train_seqs[rnd_idxs]

    batch_losses = []
    for batch_idx in range(len(train_data) // batch_size):
        # Obtain the batch data
        start = batch_idx * batch_size
        end = start + batch_size
        batch_data = train_data[start:end]
        input_data = batch_data

        # Augment the input data by adding noise
        noise = torch.bernoulli(torch.zeros_like(input_data) + 0.8)
        input_data = input_data * noise

        # Perform the training
        outputs = net(input_data)
        loss = criterion(outputs, batch_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())

    losses.append(np.mean(batch_losses))

    # Compute the accuracies for evaluation
    net.eval()

    # Compute the overall accuracy on the training dataset
    train_acc = compute_accuracy(train_data, train_resp_dicts, train_seqs)
    test_acc = compute_accuracy(test_data, test_resp_dicts, test_seqs)

    # Diagnostig output
    print('Epoch {}/{} ({:.2f}s): {}'.format(
        epoch + 1, n_epochs, time.time() - start_time, np.mean(batch_losses)))
    print('   train acc: {:.4f} ({:.4f})'.format(np.mean(train_acc), np.std(train_acc)))
    print('   test acc : {:.4f} ({:.4f})'.format(np.mean(test_acc), np.std(test_acc)))

    # Store the accuracy results
    train_accs.append(train_acc)
    test_accs.append(test_acc)

# Write the accuracies to disk
print('Writing the results to disk...')
np.save('train_accs.npy', np.array(train_accs))
np.save('test_accs.npy', np.array(test_accs))
np.save('train_losses.npy', np.array(losses))
