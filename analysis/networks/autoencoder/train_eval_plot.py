import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training results
train_accs = np.load('train_accs.npy')
test_accs = np.load('test_accs.npy')
train_losses = np.load('train_losses.npy')

plot_data = []
for acc_type, acc_data in {'Training Dataset': train_accs, 'Test Dataset': test_accs}.items():
    print(acc_data.shape)
    for epoch in range(acc_data.shape[0]):
        for subj in range(acc_data.shape[1]):
            for task in range(acc_data.shape[2]):
                plot_data.append({
                    'epoch': epoch,
                    'subj': subj,
                    'task': task,
                    'type': acc_type,
                    'value': acc_data[epoch, subj, task]
                })

plot_df = pd.DataFrame(plot_data)

# Plot
sns.set(style='whitegrid', palette='colorblind')

plt.subplot(211)
sns.lineplot(np.arange(len(train_losses)), train_losses)
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

plt.suptitle('Autoencoder')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig('train_eval_plot.pdf')
plt.show()
