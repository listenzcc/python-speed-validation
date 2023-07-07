"""
File: main.py
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
import time
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from PIL import Image
from rich import print
from tqdm.auto import tqdm
# from IPython.display import display


# %% ---- 2023-07-06 ------------------------
# Function and class

parser = argparse.ArgumentParser(description="Speed test for joblib parallel")
parser.add_argument('--parallel', '-p', type=int,
                    default=10, dest='parallel', help='Number of parallel workload, default 10')
args = parser.parse_args()
parallel = args.parallel

# print(args)
# os._exit(0)


def interpolate(src, dst, ratio):
    return (src * ratio + dst * (1-ratio)).astype(np.uint8)


# %% ---- 2023-07-06 ------------------------
# Play ground
img1 = Image.open('image/emerald_lake_in_the_andes-wallpaper-3840x2160.jpg')
img2 = Image.open('image/forza_motorsport_11-wallpaper-3840x2160.jpg')

m1 = np.array(img1)
m2 = np.array(img2)
print(m1.shape, m2.shape)


# %%
m_list = [[m1.copy(), r] for r in np.linspace(0, 1, parallel, endpoint=False)]


def test_session_1():
    tic = time.time()
    result = []

    for m, r in tqdm(m_list):
        result.append(interpolate(m, m2, r))

    toc = time.time()
    cost = toc - tic
    # print('For loop costs {} seconds'.format(cost))

    return cost, result


def test_session_2():
    tic = time.time()

    result = Parallel(
        n_jobs=-1)(delayed(interpolate)(m, m2, r) for m, r in tqdm(m_list))

    toc = time.time()
    cost = toc - tic
    # print('Joblib parallel costs {} seconds'.format(cost))

    return cost, result

# %%


costs = []

tic = time.time()
for i in tqdm(range(10), 'forLoop test'):
    cost, result = test_session_1()
    costs.append((cost, 'epoch-{}'.format(i), 'forLoop'))
toc = time.time()
costs.append((toc - tic, 'total', 'forLoop'))


tic = time.time()
for i in tqdm(range(10), 'parallel test'):
    cost, result = test_session_2()
    costs.append((cost, 'epoch-{}'.format(i), 'parallel'))
toc = time.time()
costs.append((toc - tic, 'total', 'parallel'))


# %%
df = pd.DataFrame(costs, columns=['cost', 'name', 'method'])
df['type'] = df['name'].map(lambda s: s.split('-')[0])
df['parallel'] = parallel
df.to_csv('result-parallel-{}.csv'.format(parallel))
print(df)

# input('done.')

# %%
fig, axs = plt.subplots(1, 2, figsize=(8, 4))


def setup_ax(ax, title='--'):
    ax.set_title(title)
    ax.grid(True)
    ax.set_axisbelow(True)
    return ax


ax = setup_ax(axs[0], 'Epoch compare')
sns.boxplot(df.query('type == "epoch"'), x='method',
            hue='method', y='cost', ax=ax)

ax = setup_ax(axs[1], 'Total compare')
sns.pointplot(df.query('type == "total"'), x='method',
              hue='method', y='cost',  ax=ax)


fig.suptitle('Time cost compare (10 epochs & {} parallel)'.format(parallel))
fig.tight_layout()
fig.savefig('result-parallel-{}.jpg'.format(parallel))
# plt.show()


# %% ---- 2023-07-06 ------------------------
# Pending
# display(Image.fromarray(result[2]))
# display(Image.fromarray(result[2]))

# %% ---- 2023-07-06 ------------------------
# Pending

# %%
