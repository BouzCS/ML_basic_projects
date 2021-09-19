import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
from sklearn.datasets import make_classification
warnings.filterwarnings('ignore')
%matplotlib inline

df = pd.read_csv('train.csv')
df= make_classification(random_state=42)

# print(df.columns)


# sns.distplot(df['SalePrice']);

# var = 'GrLivArea'
# data = pd.concat([df['SalePrice'], df[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

# var2 = 'TotalBsmtSF'
# data = pd.concat([df['SalePrice'], df[var2]], axis=1)
# data.plot.scatter(x=var2, y='SalePrice', ylim=(0,800000));

# var = 'OverallQual'
# data = pd.concat([df['SalePrice'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(8, 6))
# fig = sns.boxplot(x=var, y="SalePrice", data=data)
# fig.axis(ymin=0, ymax=800000);

# #correlation matrix
# corrmat = df.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True);

# #saleprice correlation matrix
# k = 10 #number of variables for heatmap
# cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(df[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()

# #scatterplot
# sns.set()
# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(df[cols], size = 2.5)
# plt.show();