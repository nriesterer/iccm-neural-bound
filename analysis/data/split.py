""" Splits the Ragni2016 dataset into a random training (N=100) and test (N=39) dataset.

"""

import pandas as pd
import numpy as np


# Load the raw data
df = pd.read_csv('Ragni2016.csv')

# Split into training and test dataset
n_train = 100
train_ids = np.random.choice(df['id'].unique(), size=n_train, replace=False)
train_df = df.loc[df['id'].isin(train_ids)]
test_df = df.loc[~df['id'].isin(train_ids)]

# Store the data
train_df.to_csv('Ragni-train.csv', index=False)
test_df.to_csv('Ragni-test.csv', index=False)
