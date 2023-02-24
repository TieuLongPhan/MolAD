import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

class PCA_conf_elip:
    
    def __init__(self, data, Type, ID, figsize, savefig = False):
        self.data = data
        self.ID = ID
        self.Type = Type
        self.figsize = figsize
        self.savefig = savefig
    
    
    def confidence_ellipse(self, x, y, ax, n_std=1, facecolor='none', **kwargs):
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*.

        Parameters
        ----------
        x, y : array-like, shape (n, )
            Input data.

        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.

        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.

        **kwargs
            Forwarded to `~matplotlib.patches.Ellipse`

        Returns
        -------
        matplotlib.patches.Ellipse
        """
        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        cov = np.cov(x, y)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensional dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                          facecolor=facecolor, **kwargs)

        # Calculating the standard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)

        # calculating the standard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)
    
    def fit(self):
        x = self.data[self.data[self.Type]=='Train'].PC1
        y = self.data[self.data[self.Type]=='Train'].PC2
        plt.figure(figsize =self.figsize)
        ax = sns.scatterplot(data=self.data, x="PC1", y="PC2", hue="Data")
        ax.axvline(c='grey', lw=1)
        ax.axhline(c='grey', lw=1)
        plt.xlabel("PC1",fontweight='bold', fontsize = 12)
        plt.ylabel("PC2",fontweight='bold', fontsize = 12)

        #confidence_ellipse(x, y, ax, edgecolor='red', n_std=3)
        self.confidence_ellipse(x, y, ax, n_std=1,
                           label=r'$1\sigma$', edgecolor='firebrick')
        self.confidence_ellipse(x, y, ax, n_std=2,
                           label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
        self.confidence_ellipse(x, y, ax, n_std=3,
                           label=r'$3\sigma$', edgecolor='blue', linestyle=':')
        ax.set_title('Applicability domain - PCA confidence ellipse', fontsize =16, weight ='semibold')
        ax.legend()
        if self.savefig == True:

            plt.savefig('Img/pca_confidence_ellipse.png', dpi = 300)
