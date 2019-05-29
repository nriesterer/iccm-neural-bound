import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if len(sys.argv) < 2:
    print('Usage: python plot_results.py <ccobra-results.csv>')
    sys.exit(99)

# Load the data and rename models
df = pd.read_csv(sys.argv[1])
df['model'] = df['model'].replace({
    'MLP-Adapt': 'MLP',
    'UniformModel': 'Random',
    'PHM': 'Probability\nHeuristics',
    'MMT': 'Mental Models\nTheory',
    'VerbalModels': 'Verbal\nModels',
    'MFAModel': 'MFA'
})

# Aggregate for models and individuals to prepare for correct error bars
df = df.groupby(['id', 'model'], as_index=False)['hit'].agg('mean')

# Determine performance ordering
order = df.groupby('model', as_index=False)['hit'].agg('mean').sort_values('hit')['model']

# Coloring
colorcodes = {
    'Random': 1,
    'MFA': 1,
    'RNN': 2,
    'MLP': 2,
    'Autoencoder': 2
}
color_palette = ['C{}'.format(colorcodes[x] if x in colorcodes else 0) for x in order]

# Plot the results
sns.set(style='whitegrid', palette='colorblind')

plt.figure(figsize=(10,3))
sns.barplot(x='model', y='hit', data=df, order=order, palette=color_palette)
sns.despine()

plt.xticks(rotation=45)
plt.yticks(np.arange(0, 0.51, 0.1))
plt.xlabel('')
plt.ylabel('Mean Predictive\nAccuracy')

plt.tight_layout()
if len(sys.argv) > 2:
    plt.savefig(sys.argv[2])
plt.show()
