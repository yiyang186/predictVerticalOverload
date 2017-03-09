from functions import *

touch_down = 480
rate = 8

workdir = "/home/pyy/data/cast/vrtg_-60_60/"
filenames = os.listdir(workdir)
fns = map(lambda fn: handle_filename(fn.split('-')), filenames)
df = pd.DataFrame(fns, columns=["INDEX", "src", "dst", "time_d", "date_d", "filename", "datetime"])
df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
del df['time_d'], df['date_d']
df['INDEX'] = df['INDEX'].astype(np.int)

'''
correct_file = "F:/Workspaces/python/cast/landing/input/lvhao_result_3.csv"
correct = pd.read_csv(correct_file, na_values=-1)
correct = correct[["INDEX", "mean"]].dropna()
correct['INDEX'] = correct['INDEX'].astype(np.int)
df =pd.merge(df, correct, how='left', on="INDEX")
df = df.dropna()
'''
vrtg_size = touch_down + rate*30 + 1
other_size = 3 # df表mergele
t1 = -3 # touch_down后t1秒， t1<0表示在touch_down之前
t2 =  3# touch_down后t2秒

vrtgs = np.zeros([df.shape[0], vrtg_size + other_size])
for i in range(df.shape[0]):
    fn = df.loc[df.index[i], 'filename']
    dd = pd.read_csv(workdir +fn, names=["index", "VRTG", "PITCH"], skiprows=1)
    if dd.shape[0] < vrtg_size:
        continue
    vrtgs[i, 0] = df.loc[df.index[i], "INDEX"]
    #vrtgs_diff = dd.loc[0: vrtg_size-1, "VRTG"] - df.loc[df.index[i], "mean"]
    vrtgs_diff = dd.loc[0: vrtg_size-1, "VRTG"] - dd.loc[touch_down+rate*10:, "VRTG"].mean()
    vrtgs[i, 1] = round(vrtgs_diff[touch_down+rate*t1: touch_down+rate*t2].max(), 3)  # vrtg_max for sort
    index_max = vrtgs_diff[touch_down+rate*t1: touch_down+rate*t2].idxmax()
    vrtgs[i, 2] = index_max
    bias = index_max - touch_down
    if bias >= 0:
        vrtgs[i, other_size: vrtgs_diff.shape[0] + other_size - bias] = vrtgs_diff[bias:]
    else:
        vrtgs[i, other_size - bias:] = vrtgs_diff[:vrtgs_diff.shape[0] + bias]
   
vrtgs_df = pd.DataFrame(vrtgs, columns=['INDEX', "MAX_VRTG", "INDEX_MAX_VRTG"] +
                        map(lambda i : str(i), range(vrtg_size)))
df =pd.merge(df, vrtgs_df, how='left', on="INDEX")
df = df.sort_values(["MAX_VRTG"], ascending=True)
df = df.dropna(axis=0)
df = df.set_index("INDEX")
max_vrtg = df.loc[:, "MAX_VRTG"].round(3).values