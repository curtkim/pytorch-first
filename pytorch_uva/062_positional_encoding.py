import numpy as np

## Imports for plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.reset_orig()

from common_transformer import PositionalEncoding

encod_block = PositionalEncoding(d_model=48, max_len=96)
print('encod_block.pe.shape', encod_block.pe.shape)
pe = encod_block.pe.squeeze().T.cpu().numpy()


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
pos = ax.imshow(pe, cmap="RdGy", extent=(1, pe.shape[1] + 1, pe.shape[0] + 1, 1))
fig.colorbar(pos, ax=ax)
ax.set_xlabel("Position in sequence")
ax.set_ylabel("Hidden dimension")
ax.set_title("Positional encoding over hidden dimensions")
ax.set_xticks([1] + [i * 10 for i in range(1, 1 + pe.shape[1] // 10)])
ax.set_yticks([1] + [i * 10 for i in range(1, 1 + pe.shape[0] // 10)])
plt.show()

sns.set_theme()
fig, ax = plt.subplots(2, 2, figsize=(12, 4))
ax = [cell for row in ax for cell in row]
for i in range(len(ax)):
    ax[i].plot(np.arange(1, 17), pe[i, :16], color=f'C{i}', marker="o", markersize=6, markeredgecolor="black")
    ax[i].set_title(f"Encoding in hidden dimension {i + 1}")
    ax[i].set_xlabel("Position in sequence", fontsize=10)
    ax[i].set_ylabel("Positional encoding", fontsize=10)
    ax[i].set_xticks(np.arange(1, 17))
    ax[i].tick_params(axis='both', which='major', labelsize=10)
    ax[i].tick_params(axis='both', which='minor', labelsize=8)
    ax[i].set_ylim(-1.2, 1.2)
fig.subplots_adjust(hspace=0.8)
sns.reset_orig()
plt.show()

