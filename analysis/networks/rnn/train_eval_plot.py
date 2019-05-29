import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
train_accs = np.load('train_accs.npy')
test_accs = np.load('test_accs.npy')
train_losses = np.load('train_losses.npy')

# Make plot data
plot_data = []
for key, arr in {'Training Dataset': train_accs, 'Test Dataset': test_accs}.items():
    for epoch in range(arr.shape[0]):
        for subj in range(arr.shape[1]):
            plot_data.append({
                'epoch': epoch,
                'subj': subj,
                'type': key,
                'value': arr[epoch, subj]
            })
plot_df = pd.DataFrame(plot_data)

# Plot
sns.set(style='whitegrid', palette='colorblind')

plt.subplot(211)
sns.lineplot(np.arange(len(train_losses)), train_losses)
plt.xlabel('')
plt.ylabel('Crossentropy Loss')
plt.ylim(bottom=0)

plt.subplot(212)
sns.lineplot(x='epoch', y='value', hue='type', data=plot_df)

handles, labels = plt.gca().get_legend_handles_labels()
plt.gca().legend(
    handles=handles[1:], labels=labels[1:], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    ncol=2, mode="expand", borderaxespad=0., frameon=False)

plt.xlabel('Epochs')
plt.ylabel('Predictive Accuracy')

plt.suptitle('RNN')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig('train_eval_plot.pdf')
plt.show()
