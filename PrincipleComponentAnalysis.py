from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from pca import pca  # https://github.com/erdogant/pca
import seaborn as sns
sns.set()
"""https://stackoverflow.com/questions/46732075/python-pca-plot-using-hotellings-t2-for-a-confidence-interval/63043840#63043840"""


file = r"C:\Users\bsa\OneDrive - Orac Holding\Desktop\DOE\food-texture.csv"
data = pd.read_csv(file, index_col=[0], decimal='.', sep=',')
# print(data)
# plt.plot(data)

'''by the book nu, scalen, fitten, transformen?'''
x = StandardScaler().fit_transform(data)
print(pd.DataFrame(x))
# x = data.values
plt.figure()
plt.boxplot(x)  # ziet er ok uit
plt.gca().set(title='Scaled values')
model = PCA(svd_solver='full').fit(x)

plt.figure()
plt.plot(np.cumsum(model.explained_variance_ratio_), color='red')
plt.bar(np.arange(model.n_components_), model.explained_variance_ratio_)
plt.gca().set(title='95%')
print('components waarmee t-waarde wordt berekend:\n', pd.DataFrame(model.components_.T, index=data.columns))

plt.figure()
plt.subplot(211)
plt.bar(data.columns, model.components_[:, 0])
plt.gca().set(title='PC1', ylabel='first component loading')
plt.subplot(212)
plt.bar(data.columns, model.components_[:, 1])
plt.gca().set(title='PC2', ylabel='second component loading')
plt.tight_layout()

plt.figure()
plotpca = model.transform(x)
plt.scatter(x=plotpca[:, 0], y=plotpca[:, 1])
for i, txt in enumerate(data.index):
    plt.annotate(i, (plotpca[:, 0][i], plotpca[:, 1][i]))
plt.gca().set(title='scores', xlabel='first component', ylabel='second component')

plt.figure()
plt.plot(plotpca[:, 0])
print(plotpca[:, 0][32])
plt.gca().set(title='scores van PC1')

plt.figure()
plt.scatter(x=model.components_[:, 0], y=model.components_[:, 1])
for i, txt in enumerate(data.columns):
    plt.annotate(txt, (model.components_[:, 0][i], model.components_[:, 1][i]))
plt.gca().set(xlabel='first component', ylabel='second component')
plt.show()

print(data.reset_index())
print(data.mean())




'''
Dit werkt kinda, maar ik ben niet zeker of mijn data wel ok gesscaled wordt
'''
# x = data.values
# model = pca()
# out = model.fit_transform(x)
# print('\n', out['topfeat'], '\n')
# print('% dat Principle component bijdraagd om 95% van de waarden te verklaren (telt op tot 1)'
#       ' \n', out['explained_var'][:10])
# model.plot(n_components=4)
# model.biplot(n_feat=4, legend=True, SPE=True, hotellingt2=True)  # hotellingt2 = CHI kwadraat test
# model.biplot3d(n_feat=4, legend=True, SPE=True, hotellingt2=True)
# non_outliers = x[~out['outliers']['y_bool'], :]

'''hier probeer ik via PCA te transformeren maar het werkt niet'''
# plt.figure()
# p = PCA(n_components=4).fit(data)
# p_p = p.transform(data)
# print(p_p)
# plt.boxplot(p_p[:,0])

