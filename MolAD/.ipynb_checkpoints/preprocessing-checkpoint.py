
import sys
sys.path.append('ultility')
from Featurizer import Featurize

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



class prepare_dataset:
    
    def __init__(self, data_train, data_test, smile_col, mol_col, ID, activity_col, feature_col, fp_type):
        self.data_train = data_train
        self.data_test = data_test
        self.smile_col = smile_col
        self.mol_col = mol_col
        self.activity_col = activity_col
        self.ID = ID
        self.feature_col = feature_col
        self.fp_type = fp_type # RDKFp;  ECFPs; MACCs
        self.feature = Featurize(data = self.data_train, smile_col =self.smile_col,
                            activity_col=self.activity_col, ID = self.ID, save_dir = None, m2v_path = None)
        
    def fp_call(self): 
        if self.feature_col ==  None:
            if self.fp_type == 'ECFPs':

                fp = self.data_train[self.mol_col].apply(self.feature.ECFPs)
                X = np.stack(fp.values)
                self.train = pd.DataFrame(X)

                fp = self.data_test[self.mol_col].apply(self.feature.ECFPs)
                X = np.stack(fp.values)
                self.test= pd.DataFrame(X)
        else:
            self.train = self.data_train[self.feature_col]
            self.test = self.data_test[self.feature_col]
            
    def pca_reduce(self):
        pca = PCA(n_components=2)
        pca.fit(self.train)

        x_train = pca.transform(self.train)
        x_test = pca.transform(self.test)
        
        df_1 = pd.DataFrame(x_train)
        df_1['Data'] = 'Train'
        df_1['ID'] = self.data_train[self.ID]
        
        df_2 = pd.DataFrame(x_test)
        df_2['Data'] = 'Test'
        df_2['ID'] = self.data_test[self.ID]
        
        self.df_pca = pd.concat([df_1,df_2], axis = 0).reset_index(drop=True)
        self.df_pca.columns = ['PC1','PC2','Data', 'ID']
        
    def fit(self):
        self.fp_call()
        self.pca_reduce()
