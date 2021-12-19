import os
from lab3_task2 import *
from tqdm import tqdm

directory = 'saved/album-4086_283220248'
filenames = []
data = []
for filename in tqdm(os.listdir(directory)):
    data_iter = []
    filenames.append(filename)
    img = str(directory) + '\\' + filename
    clusters = 2
    dc = DominantColors(img, clusters)
    colors = dc.dominantColors()
    for i in colors:
        for j in i:
            data_iter.append(j)
    data.append(data_iter)

np_data = np.asarray(data, dtype=np.float32)
print(np_data)
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
XPCAreduced = pca.fit_transform(np_data)
print (XPCAreduced)
print(filenames)

xs, ys, zs = np_data[:, 0], np_data[:, 1], np_data[:,2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()