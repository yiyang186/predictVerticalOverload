from function import *
from sklearn.decomposition import PCA

needcols = ['DRIFT', 'EST_SID_SLIP', 'VRTG', 'ROLL_CPT', 'AOAL', 'AOAR', 'ROLL', 'RUDD', 'RUD_PDL', 'VAPP', 'ELEVL',
            'ELEVR', 'AIL_INBL', 'AIL_INBR', 'AIL_OUBL', 'AIL_OUBR', 'LATG', 'WIN_SPDR', 'WIN_DIR', 'HEAD_MAG']

todo = ['DRIFT', 'EST_SID_SLIP', 'ROLL_CPT_diff', 'DRIFT_diff', 'VRTG', 'ROLL_CPT', 'AOAL_diff', 'AOAR_diff', 'ROLL_diff',
        'RUDD_diff', 'ROLL', 'RUD_PDL', 'VAPP', 'ELEVL_diff', 'ELEVR_diff', 'AIL_INBL', 'AIL_INBR', 'AIL_OUBL', 'AIL_OUBR',
        'AIL_INBL_diff', 'AIL_INBR_diff', 'AIL_OUBL_diff', 'AIL_OUBR_diff', 'RUD_PDL_diff', 'RUDD', 'LATG', 'WIN_X', 'WIN_Y']
n_components = 5

for i, index in enumerate(df.index):
    fn = workdir2 + df.loc[index, "filename"]
    ed = int(df.loc[index, "INDEX_MAX_VRTG"] - 1)
    if not os.path.exists(fn):
        continue
    fn_df = pd.read_csv(fn, usecols=needcols)
    matrix = np.zeros((50, len(todo)))
    for j, f in enumerate(todo):
        if j > len(todo) - 3:
            break
        if '_diff' in f:
            col = f[: -5]
            rate = params[col]
            data = np.diff(fn_df.loc[: ed, col].dropna().values[-51*rate: ].reshape((51, rate))[: , 0])
        else:
            rate = params[f]
            data = fn_df.loc[: ed, f].dropna().values[-50*rate: ].reshape((50, rate))[: , 0]
        matrix[:, j] = data
    win_spdr = fn_df.loc[: ed, 'WIN_SPDR'].dropna().values[-50:]
    win_dir = fn_df.loc[: ed, 'WIN_DIR'].dropna().values[-50:]
    win_hmg = fn_df.loc[: ed, 'HEAD_MAG'].dropna().values[-50:]
    win_x, win_y = composite_wind(win_spdr, win_dir, win_hmg)
    matrix[:, -2] = win_x
    matrix[:, -1] = win_y
    fn_pca = PCA(n_components=5)
    new_matrix = fn_pca.fit_transform(matrix)
    np.savetxt("/home/pyy/data/cast/5_pca/"+df.loc[index, "filename"], new_matrix, fmt='%.4f', delimiter=',')
    if i % 100 == 0:
        print sum(fn_pca.explained_variance_ratio_)