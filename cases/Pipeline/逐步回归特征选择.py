# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:32:29 2019

@author: atlan
"""

from time import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier


#Random Seed
seed = np.random.randint(0, 10000)
print('Seed:', seed)
np.random.seed(seed)

def stepwise_FeSel(data):
    """
    前向逐步回归特征选择
    """
    fixed = []    
    y = data.pop('分组')
    X = data
    print(X.shape)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)

    remaining = set(X.columns).difference(set(fixed))
#    remaining = ['红细胞计数', '总胆红素']
    num_remaining = len(remaining)
    selected = fixed.copy()
    model = GradientBoostingClassifier(random_state=seed)
    if len(fixed) == 0:
        current_score = 0
    else:
        current_score = cross_val_score(model, X[fixed], y, cv=cv).mean()
    best_new_score = current_score
    n = 1
    scores = [['', best_new_score]]
    Feature_score = []
    while remaining and current_score == best_new_score:
        time0 = time()
        print('%d/%d, score: %f' % (n, num_remaining, current_score))
        scores_with_candidates = []
        for candidate in tqdm(remaining):
            features = selected + [candidate]
            X_ = X[features]
            score = cross_val_score(model, X_, y, cv=cv, n_jobs=2).mean()
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if best_new_score <= current_score:
            break
        elif current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print('select %s, score: %f, elapsed time: %f' %
                  (best_candidate, current_score, time()-time0))
            scores.append([best_candidate, best_new_score])

        n += 1
    pd.DataFrame(Feature_score, columns=['特征','train','valid','test']).to_excel('特征score-%d.xlsx' % seed, index=False)
    """
    特征重要性
    """
    X_ = X[selected]
    model.fit(X_, y)
    FeaIm = model.feature_importances_
    FeaIm = pd.DataFrame({'特征': selected, '重要性': FeaIm.tolist()})
    scores = pd.DataFrame(scores)
    scores = pd.concat([scores, FeaIm], axis=1)
    pd.DataFrame(scores).to_excel('Forward_feature_select_1213_%d.xlsx' % (seed), index=False)

def main():
    train = pd.read_excel('train.xlsx')
    test = pd.read_excel('test.xlsx')
    data = pd.concat([train, test], axis=0)
    stepwise_FeSel(data)
  