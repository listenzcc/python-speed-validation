"""
File: batch.py
Author: Chuncheng Zhang
Date: 2023-07-06
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Amazing things

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2023-07-06 ------------------------
# Requirements and constants
import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rich import print


# %% ---- 2023-07-06 ------------------------
# Function and class

parser = argparse.ArgumentParser(
    description='Batch script for speed testing, using main.py to compute the benchmark')
parser.add_argument('-e', '--re-run', action='store_true',
                    help='Whether re-run the main.py script, if not set, only summary the results, if')
args = parser.parse_args()

print(args)
# os._exit(0)


if args.re_run:
    for p in [10, 20, 40, 60]:
        os.system('python main.py -p {}'.format(p))
        pass


# %% ---- 2023-07-06 ------------------------
# Play ground
dfs = []
for p in [10, 20, 40, 60]:
    dfs.append(pd.read_csv('result-parallel-{}.csv'.format(p), index_col=0))

df = pd.concat(dfs)
print(df)

totals = df.query('type=="total"')
print(totals)
group = totals.groupby(by='parallel')
ratio_table = group.max('cost') / group.min('cost')
ratio_table['parallel'] = ratio_table.index
ratio_table['speedup'] = ratio_table['cost']
print(ratio_table)

# %% ---- 2023-07-06 ------------------------
# Pending
fig, axs = plt.subplots(1, 3, figsize=(10, 4))

# loc = 'lower right'


def setup_ax(ax, title='--'):
    ax.set_title(title)
    ax.grid(True)
    ax.set_axisbelow(True)
    return ax


ax = setup_ax(axs[0], 'Epoch compare')
sns.boxplot(df.query('type == "epoch"'), x='method',
            hue='parallel', y='cost', ax=ax)
ax.legend(loc='upper left')

ax = setup_ax(axs[1], 'Total compare')
sns.pointplot(df.query('type == "total"'), x='method',
              hue='parallel', y='cost', ax=ax)
ax.legend(loc='upper right')

ax = setup_ax(axs[2], 'Speedup by choosing method')
sns.pointplot(ratio_table, x='parallel', hue='parallel', y='speedup', ax=ax)
ax.legend(loc='upper right')

fig.suptitle('Time cost compare (10 epochs summary)')
fig.tight_layout()
fig.savefig('summary.jpg')


# %% ---- 2023-07-06 ------------------------
# Pending


# %%

# %%
