# coding:utf-8

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# use TkAgg backend
matplotlib.use("TkAgg")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 180
plt.rcParams['figure.dpi'] = 180

stat_folder = './output/eda/'
if not os.path.exists(stat_folder):
    os.makedirs(stat_folder)


def stat_missing_values(data):
    """
    统计缺失值数量，并画图，保存为"缺失值统计.png"
    :param data: dataframe, 需要统计的缺失值
    :return: plt
    """
    missing = data.isnull().sum(axis=0).reset_index()
    missing.columns = ['column_name', 'missing_count']
    missing = missing.loc[missing['missing_count'] > 0]
    missing = missing.sort_values(by='missing_count')

    ind = np.arange(missing.shape[0])
    fig, ax = plt.subplots(figsize=(12, 18))
    rects = ax.barh(ind, missing.missing_count.values)
    cnt = missing.missing_count.values
    for j, rect in enumerate(rects):
        plt.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height() / 2,
                 '%.0f' % cnt[j], ha='center', va='center', fontsize=12)

    ax.set_yticks(ind)
    ax.set_yticklabels(missing.column_name.values, rotation='horizontal')
    ax.set_xlabel("Count of missing values")
    ax.set_title("Number of missing values")
    plt.savefig(stat_folder + "缺失值统计.png")
    return plt


def stat_freq_continious(data, num_bins=10):
    """
    统计连续型特征的频次，data的列必须是连续型。
    :param data: DataFrame
    :param num_bins:
    :return:
    """
    values = data.columns.values
    print(values)
    data = data.fillna('nan')
    # 连续性变量，等宽成num_bins来处理
    cut_bins = np.linspace(0, 1, num_bins + 1)
    q = data.quantile(cut_bins).values
    length = len(values)
    for i in range(length):
        name = values[i]
        # 合并相同区间（幂律分布可能导致很多分位数值相同）
        q_this = list(set(q[:, i]))
        q_this.sort()
        if len(q_this) > 1:
            # cut是左开区间，所以将0分位数减去epsilon
            q_this[0] = q_this[0] - np.finfo(np.float32).eps
            cut_labels = ["%.2f" % float(p) + "~" + "%.2f" % float(n) for (p, n) in list(zip(q_this, q_this[1:]))]
            data[name + '_cut'] = pd.cut(data[name], bins=q_this, labels=cut_labels)
            counts = data[name + '_cut'].value_counts(dropna=False)
        else:
            # 如果该col的值总共只有1个
            counts = data[name].value_counts(dropna=False)
        indexes = counts.index.values
        values = counts.values

        # 按照区间值排序
        def sort_func(d):
            try:
                return float(d[0].split("~")[0])
            except AttributeError:
                return float(np.inf)

        lst = sorted(zip(indexes, values), key=sort_func)
        x, y = list(zip(*lst))
        plt.bar(x, y)
        plt.xticks(rotation=75)
        plt.subplots_adjust(bottom=.3)
        plt.title(name)
        plt.ylabel('频次')
        # plt.show()
        plt.savefig(stat_folder + "频次-" + name + ".png")
        plt.gcf().clear()


def stat_freq_categrical(data):
    """
    统计离散型特征的频次
    :param data: DataFrame
    :return:
    """
    values = data.columns.values
    features = data.fillna('nan')
    # 连续性变量，等宽成num_bins来处理
    length = len(values)
    for i in range(length):
        name = values[i]
        counts = features[name].value_counts(dropna=False)
        counts.plot(kind='bar')
        plt.xticks(rotation=75)
        plt.subplots_adjust(bottom=.3)
        plt.title(name)
        plt.ylabel('频次')
        # plt.show()
        plt.savefig(stat_folder + "频次-" + name + ".png")
        plt.gcf().clear()


def stat_freq(data, data_type='dataframe', names=None, force_cat_cols=None, num_bins=10):
    """
    统计频率，每次处理的categorical类型是一样的
    :param data: 需处理的数据
    :param data_type: dataframe 或者 ndarray
    :param names: 列名列表
    :param force_cat_cols: 强制指定列名为离散型变量；默认为None，则按照是否为数字来自动划分categorical/continuous。
    :param num_bins: 等分的bin数目
    :return: 生成图片文件到./eda/stat/目录下
    """
    if data_type == "dataframe":
        features = data
        names = features.columns.values
    else:
        if names is None:
            raise ValueError("names must be set")
        if data is None:
            raise ValueError("data must be set")
        dim = np.ndim(data)
        if dim > 2:
            raise ValueError("only less than 3 dimension is supported")
        num_cols = 1 if dim == 1 else data.shape[-1]
        if num_cols != len(names):
            raise ValueError("the length of names must be equal the length of data")
        features = pd.DataFrame(data, columns=names)

    df_tmp = features.dropna(axis=1, how='all').apply(lambda xx: pd.to_numeric(xx, errors='ignore'))
    df_tmp = df_tmp.select_dtypes(include=[np.number])
    continuous_cols = df_tmp.columns.tolist()

    features = features.fillna('nan')
    names_cat = []
    for item in names:
        # 1. continuous_cols为空
        # 2. col不在continuous_cols中
        # 3. 强制设为cat
        if not continuous_cols or item not in continuous_cols or (force_cat_cols and item in force_cat_cols):
            names_cat.append((1, item))
        else:
            names_cat.append((0, item))
    if continuous_cols:
        # 连续性变量，等宽成num_bins来处理
        cut_bins = np.linspace(0, 1, num_bins + 1)
        q = features[continuous_cols].quantile(cut_bins).values
    length = len(names_cat)
    for i in range(length):
        name = names_cat[i][1]
        categorical = names_cat[i][0]
        if categorical:
            counts = features[name].value_counts(dropna=False)
            counts.plot(kind='bar')
        else:
            # 合并相同区间（幂律分布可能导致很多分位数值相同）
            q_this = list(set(q[:, continuous_cols.index(name)]))
            q_this.sort()
            if len(q_this) > 1:
                # cut是左开区间，所以将0分位数减去epsilon
                q_this[0] = q_this[0] - np.finfo(np.float32).eps
                cut_labels = ["%.2f" % float(p) + "~" + "%.2f" % float(n) for (p, n) in list(zip(q_this, q_this[1:]))]
                features[name + '_cut'] = pd.cut(features[name], bins=q_this, labels=cut_labels)
                counts = features[name + '_cut'].value_counts(dropna=False)
            else:
                # 如果该col的值总共只有1个
                counts = features[name].value_counts(dropna=False)
            indexes = counts.index.values
            values = counts.values

            # 按照区间值排序
            def sort_func(d):
                try:
                    return float(d[0].split("~")[0])
                except AttributeError:
                    return float(np.inf)

            lst = sorted(zip(indexes, values), key=sort_func)
            x, y = list(zip(*lst))
            plt.bar(x, y)
        plt.xticks(rotation=75)
        plt.subplots_adjust(bottom=.3)
        plt.title(name)
        plt.ylabel('频次')
        # plt.show()
        plt.savefig(stat_folder + "频次-" + name + ".png")
        plt.gcf().clear()


def stat_dist(x, y=None, data_type='dataframe', feature_names=None, force_cat_cols=None, num_bins=10):
    """
    绘制属性-标签分布图
    :param x: dataframe或者ndarray, 属性数据
    :param y: 标签数据
    :param feature_names: 属性名列表
    :param force_cat_cols: 强制指定列名为离散型变量；默认为None，则按照是否为数字来自动划分categorical/continuous。
    :param num_bins: 等分的bin数目
    :return: 生成图片文件到./eda/stat/目录下
    """
    from collections import defaultdict
    if data_type != 'dataframe':
        if np.ndim(y) != 1:
            raise ValueError("only single label is supported")
        label = pd.DataFrame(y, columns=["label"])

        num_cols = 1 if np.ndim(x) == 1 else x.shape[-1]
        if num_cols != len(feature_names):
            raise ValueError("the length of x must be equal the length of feature_names")

        # 填充默认值"nan"
        features = pd.DataFrame(x, columns=feature_names)
        df = pd.concat([label, features], axis=1)
    else:
        df = x
        features = df[df.columns.difference(['label'])]
        feature_names = features.columns.values

    df_tmp = features.dropna(axis=1, how='all').apply(lambda xx: pd.to_numeric(xx, errors='ignore'))
    continuous_cols = df_tmp.select_dtypes(include=[np.number]).columns.tolist()

    # fillna
    df = df.fillna('nan')

    names_cat = []
    for item in feature_names:
        if not continuous_cols or item not in continuous_cols or (force_cat_cols and item in force_cat_cols):
            names_cat.append((1, item))
        else:
            names_cat.append((0, item))
    for idx in range(len(feature_names)):
        # feature = feature_names[idx]
        feature = names_cat[idx][1]
        categorical = names_cat[idx][0]
        if categorical:
            counts = df.groupby(["label", feature]).size().to_frame().unstack(fill_value=0).stack().to_records()
            dct_labels = defaultdict(list)
            for item in counts:
                dct_labels[item[0]].append((item[1], item[2]))
        else:
            cut_bins = np.linspace(0, 1, num_bins + 1)
            q = features.loc[:, continuous_cols].quantile(cut_bins).values
            q_this = list(set(q[:, continuous_cols.index(feature)]))
            q_this.sort()
            if len(q_this) > 1:
                # 如果该col的值总共有多个
                q_this[0] = q_this[0] - np.finfo(np.float32).eps

                # 按照分位数进行cut
                cut_labels = ["%.2f" % float(p) + "~" + "%.2f" % float(n) for (p, n) in list(zip(q_this, q_this[1:]))]
                df[str(feature) + '_cut'] = pd.cut(df[feature], bins=q_this, labels=cut_labels)
                counts = df.groupby(["label", feature + '_cut']).size().to_frame().unstack(
                    fill_value=0).stack().to_records()
            else:
                # 如果该col的值总共只有1个
                counts = df.groupby(["label", feature]).size().to_frame().unstack(
                    fill_value=0).stack().to_records()
            dct_labels = defaultdict(list)
            for item in counts:
                # {label值：[(属性值， label个数),]}
                dct_labels[item[0]].append((item[1], item[2]))

        # 实现stacked功能
        pre_cnt = None
        for key in dct_labels.keys():
            value = np.array(dct_labels[key])
            attr = np.array([item[0] for item in value])
            cnt = np.array([item[1] for item in value]).astype(np.float32)
            if pre_cnt is None:
                container = plt.bar(attr, cnt, align='center')
            else:
                container = plt.bar(attr, cnt, align='center', bottom=pre_cnt)
            if pre_cnt is None:
                pre_cnt = cnt
            else:
                pre_cnt += cnt
            for j in range(len(container.patches)):
                rect = container.patches[j]
                if cnt[j] > 0:
                    plt.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height() / 2,
                             '%.0f' % cnt[j], ha='center', va='bottom', fontsize=12)
        plt.title(feature + "-标签 分布图")
        plt.xticks(rotation=75)
        plt.subplots_adjust(bottom=.3)
        plt.legend(dct_labels.keys())
        plt.ylabel('频次')
        plt.savefig(stat_folder + "分布图-" + feature + "-标签.png")
        # plt.show()
        plt.gcf().clear()


def custom_sort(df, column_idx, key):
    """
    对dataframe按照某col的某种方式key进行排序
    :param df: dataframe
    :param column_idx: 按照排序的col列
    :param key: 排序的方式
    :return: 排序后dataframe
    """
    digit = str(column_idx).isdigit()
    if digit:
        col = df.iloc[:, column_idx]
    else:
        col = df.loc[:, column_idx]
    tmp = np.array(col.values.tolist())
    order = sorted(range(len(tmp)), key=lambda j: key(tmp[j]))
    return df.iloc[order]
