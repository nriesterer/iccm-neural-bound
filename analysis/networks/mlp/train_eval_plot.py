import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
train_accs = np.load('train_accs.npy')
train_accs_adapt = np.load('train_accs_adapt.npy')
test_accs = np.load('test_accs.npy')
test_accs_adapt = np.load('test_accs_adapt.npy')
train_losses = np.load('train_losses.npy')

data_dict = {
    # 'Training Data': train_accs,
    'Training Data': train_accs_adapt,
    # 'Test Data': test_accs,
    'Test Data': test_accs_adapt
}

# Convert to pandas
plot_data = []
for data_type, arr in data_dict.items():
    for epoch in range(arr.shape[0]):
        for subj in range(arr.shape[1]):
            for task in range(arr.shape[2]):
                plot_data.append({
                    'epoch': epoch,
                    'subj': subj,
                    'task': task,
                    'type': data_type,
                    'value': arr[epoch, subj, task]
                })

plot_df = pd.DataFrame(plot_data)

# Plot the results
sns.set(style='whitegrid', palette='colorblind')

plt.subplot(211)
sns.lineplot(np.arange(len(train_losses)), train_losses)
plt.xlabel('')
plt.ylabel('Mean Squared Error')
plt.ylim(bottom=0)

plt.subplot(212)
sns.lineplot(x='epoch', y='value', hue='type', data=plot_df)

handles, labels = plt.gca().get_legend_handles_labels()
plt.gca().legend(
    handles=handles[1:], labels=labels[1:], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    ncol=2, mode="expand", borderaxespad=0., frameon=False)

plt.xlabel('Epochs')
plt.ylabel('Predictive Accuracy')

plt.suptitle('MLP')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig('train_eval_plot.pdf')
plt.show()
