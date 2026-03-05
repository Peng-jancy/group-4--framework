from sklearn.model_selection import KFold
import numpy as np

def k_fold_cv(X,y,k=5):

    kf=KFold(n_splits=k,shuffle=True,random_state=42)

    for fold,(train_idx,val_idx) in enumerate(kf.split(X)):

        print("Fold",fold+1)