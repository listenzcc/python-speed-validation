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
import torch
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

parallel = 3
parser = argparse.ArgumentParser(description="Speed test for joblib parallel")
parser.add_argument('--parallel', '-p', type=int,
                    default=10, dest='parallel', help='Number of parallel workload, default 10')
args = parser.parse_args()
parallel = args.parallel

# print(args)
# os._exit(0)


def interpolate(src, dst, ratio, use_cuda_flag=False):
    if use_cuda_flag:
        src = torch.Tensor(src).cuda()
        dst = torch.Tensor(dst).cuda()
        return (src * ratio + dst * (1-ratio)).cpu().numpy().astype(np.uint8)

    else:
        return (src * ratio + dst * (1-ratio)).astype(np.uint8)


def interpolate_cuda(src, dst, ratio):
    return (src * ratio + dst * (1-ratio)).type(torch.uint8)


# %% ---- 2023-07-06 ------------------------
# Play ground
img1 = Image.open('image/emerald_lake_in_the_andes-wallpaper-3840x2160.jpg')
img2 = Image.open('image/forza_motorsport_11-wallpaper-3840x2160.jpg')

m1 = np.array(img1)
m2 = np.array(img2)
print(m1.shape, m2.shape)

img = Image.fromarray((m1/2+m2/2).astype(np.uint8))
img

# %%
m_list = [[m1.copy(), r] for r in np.linspace(0, 1, parallel, endpoint=False)]

m_list_cuda = [[torch.Tensor(m1).cuda(), r]
               for r in np.linspace(0, 1, parallel, endpoint=False)]
m2_cuda = torch.Tensor(m2).cuda()


def test_session_3():
    tic = time.time()

    result = []

    for m, r in tqdm(m_list_cuda):
        result.append(interpolate_cuda(m, m2_cuda, r))

    toc = time.time()
    cost = toc - tic
    # print('Cuda costs {} seconds'.format(cost))

    return cost, result


def test_session_1(use_cuda_flag):
    tic = time.time()
    result = []

    for m, r in tqdm(m_list):
        result.append(interpolate(m, m2, r, use_cuda_flag))

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

# ----------------------------------------------------------------
tic = time.time()
for i in tqdm(range(10), 'forLoop test GPU'):
    cost, result = test_session_1(use_cuda_flag=True)
    costs.append((cost, 'epoch-{}'.format(i), 'GPU'))
toc = time.time()
costs.append((toc - tic, 'total', 'GPU'))

# ----------------------------------------------------------------
tic = time.time()
for i in tqdm(range(10), 'forLoop test CPU'):
    cost, result = test_session_1(use_cuda_flag=False)
    costs.append((cost, 'epoch-{}'.format(i), 'CPU'))
toc = time.time()
costs.append((toc - tic, 'total', 'CPU'))

# ----------------------------------------------------------------
tic = time.time()
for i in tqdm(range(10), 'parallel test'):
    cost, result = test_session_2()
    costs.append((cost, 'epoch-{}'.format(i), 'parallel'))
toc = time.time()
costs.append((toc - tic, 'total', 'parallel'))

# ----------------------------------------------------------------
tic = time.time()
for i in tqdm(range(10), 'cuda test'):
    cost, result = test_session_3()
    costs.append((cost, 'epoch-{}'.format(i), 'cuda'))
toc = time.time()
costs.append((toc - tic, 'total', 'cuda'))

# %%
df = pd.DataFrame(costs, columns=['cost', 'name', 'method'])
df['type'] = df['name'].map(lambda s: s.split('-')[0])
df['parallel'] = parallel
df.to_csv('result-parallel-{}.csv'.format(parallel))
print(df)

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

# %% ---- 2023-07-06 ------------------------
# Pending

# %%
