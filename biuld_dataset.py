from functions import *

workdir3 = '/home/pyy/data/cast/5_pca/'
output = np.zeros((len(os.listdir(workdir3)), 5*5*2+2))
for i, f in enumerate(os.listdir(workdir3)):
    if f == '5_pca_features.csv':
        continue
    output[i, 0] = index = int(f[:5])
    fdf = np.loadtxt(workdir3 + f, delimiter=",")
    fdf = fdf.reshape(5, fdf.shape[0] / 5, fdf.shape[1])
    output[i, 1: 1+ 5*5] = fdf.mean(axis=1).round(4).flatten()
    output[i, 1+ 5*5: 1+ 5*5*2] = fdf.std(axis=1).round(4).flatten()
    output[i, 1+ 5*5*2 ] = df.loc[index, 'MAX_VRTG']
columns = ['P{0}_{1}_{2}'.format(i, j, k) for k in 'ms' for j in range(1, 6) for i in range(1, 6)]
columns = ['index'] + columns + ['m_vrtg']
output = pd.DataFrame(output, columns=columns)
output.to_csv(workdir3+'5_pca_features.csv')