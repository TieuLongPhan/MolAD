import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import  AllChem
from rdkit import Chem, DataStructs 
import warnings
warnings.filterwarnings("ignore")

class similarity_matrix:
    def __init__(self, data_train, data_test, ID, mol_col, function):
        self.data_train = data_train
        self.data_test = data_test
        self.ID = ID
        self.mol_col = mol_col
        self.function = function
    
    def train_process(self):
        self.list_training_name = []
        self.list_training_fp=[]
        for trainnames in self.data_train[self.ID]:
            self.list_training_name.append(trainnames)
        for mol in self.data_train[self.mol_col]:
            fgp= AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            self.list_training_fp.append(fgp)
            
    def test_process(self):
        self.list_test_name = []
        self.list_test_fp=[]
        for testnames in self.data_test[self.ID]:
            self.list_test_name.append(testnames)
        for mol in self.data_test[self.mol_col]:
            fgp= AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            self.list_test_fp.append(fgp)
    
    def fit(self):
        self.train_process()
        self.test_process()
        self.list_data_set=self.list_training_name+self.list_test_name #all data set-> training+test+external
        self.list_data_set_fp=self.list_training_fp+self.list_test_fp #all data set-> training+test+external
        
        
        size=len(self.list_data_set_fp)
        self.matrix=pd.DataFrame()
        for m, i in enumerate(self.list_data_set_fp):
            for n, j in enumerate(self.list_data_set_fp):
                similarity=DataStructs.TanimotoSimilarity(i,j)
                self.matrix.loc[self.list_data_set[m],self.list_data_set[n]]=similarity
