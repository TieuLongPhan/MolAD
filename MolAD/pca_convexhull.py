import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

class PCA_convexhull:
    
    def __init__(self, data, Type, ID, figsize, savefig = False):
        self.data = data
        self.ID = ID
        self.Type = Type
        self.figsize = figsize
        self.savefig = savefig
    
    def point_in_hull(self, point, hull, tolerance=1e-12):
            return all(
                (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
                for eq in hull.equations)
        
    def AD_PCA(self):
        self.data_train = self.data[self.data[self.Type]=='Train'].drop([self.Type, self.ID], axis = 1).values
        self.data_test = self.data[self.data[self.Type]=='Test'].drop([self.Type], axis = 1)
        self.hull = ConvexHull(self.data_train)

        in_out = []
        for p in self.data_test.drop([self.ID], axis = 1).values:
            point_is_in_hull = self.point_in_hull(point = p, hull=self.hull)
            in_out.append(point_is_in_hull)

        self.df_test = self.data_test
        self.df_test['convex'] = in_out
        

    def Visualize(self):
        sns.set()
        plt.figure(figsize =self.figsize)
        ax = sns.scatterplot(data=self.data, x="PC1", y="PC2", hue="Data")
        ax.set_title('Applicability domain - PCA convex hull', fontsize =16, weight ='semibold')
        ax.set_xlabel('PC1', weight='bold', fontsize = 12)
        ax.set_ylabel('PC2', weight='bold', fontsize = 12)


        for simplex in self.hull.simplices:
            plt.plot(self.data_train[simplex, 0], self.data_train[simplex, 1], 'k-')
            
        if self.savefig == True:

            plt.savefig('Img/pca_convex_hull.png', dpi = 300)
    def convexhull_fit(self):
        self.AD_PCA()
        self.Visualize()
