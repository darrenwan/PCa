#coding:utf-8

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


# use TkAgg backend
matplotlib.use("TkAgg")

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


# def stat_labels(y, name='label', categorical=True, num_bins=10):
#     if categorical:
#         labels = pd.Series(y)
#         counts = labels.value_counts()
#         counts.plot(kind='bar')
#         x_labels = list(counts.index.values)
#         plt.xticks(np.arange(len(x_labels)), x_labels, rotation=75)
#         plt.xlabel(name)
#         plt.ylabel('count')
#         plt.show()
#     else:
#         pass


def stat_freq(x, names=None, categorical=False, num_bins=10):
    eda_floder = './eda/'
    stat_floder = eda_floder + 'stat/'
    if not os.path.exists(eda_floder):
        os.makedirs(eda_floder)
        if not os.path.exists(stat_floder):
            os.makedirs(stat_floder)
    features = pd.DataFrame(x, columns=names)
    if not categorical:
        cut_bins = np.linspace(0, 1, num_bins + 1)
        q = features.quantile(cut_bins).values
    length = len(names)
    for i in range(length):
        name = names[i]
        if categorical:
            counts = features[name].value_counts()
            counts.plot(kind='bar', rot=75)
        else:
            q_this = list(set(q[:, i]))
            q_this.sort()
            cut_labels = ["%.2f" % float(p) + "~" + "%.2f" % float(n) for (p, n) in list(zip(q_this, q_this[1:]))]
            features[str(name) + '_cut'] = pd.cut(features[name], bins=q_this, labels=cut_labels)
            # counts = features.groupby(str(name) + '_cut').count()
            counts = features[str(name) + '_cut'].value_counts().sort_index()
            counts.plot(kind='bar', rot=75)
        plt.subplots_adjust(bottom=.3)
        plt.title(name)
        plt.ylabel('count')
        plt.savefig(stat_floder + name + ".jpg")


def stat_dist(x, y, categorical=False):
    pass

