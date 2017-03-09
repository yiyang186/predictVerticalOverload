# encode: utf-8

from functions import *

pca5_df = pd.read_csv(workdir3+'5_pca_features.csv')
pca5_df['index'] = pca5_df['index'].astype(np.int)
#pca5_df = pca5_df.set_index('index')
del pca5_df['Unnamed: 0']

pca5_df["hard"] = 0
pca5_df.loc[pca5_df['m_vrtg'] >= 0.4, 'hard'] = 1

pca5_df["weight"] = 0
pca5_df.loc[pca5_df['hard'] == 0,"weight"] = w_light = pca5_df["hard"].sum() / float(pca5_df.shape[0])
pca5_df.loc[pca5_df['hard'] == 1,"weight"]= w_hard = 1 - w_light

from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold
from sklearn.svm import SVC

def cross_validation_SVC(t, k, C, gamma):
    kf = KFold(pca5_df.shape[0], n_folds=5)
    ineed = filter(lambda c: c[3] in t, pca5_df.columns) + ['hard', 'weight']
    cnf_matrix = np.zeros((2,2))
    for train_index, valid_index in kf:
        train = pca5_df.loc[train_index, ineed]
        valid = pca5_df.loc[valid_index, ineed]
        clf = SVC(C=C, kernel='rbf', gamma=gamma)
        clf.fit(train[ineed[:-2]], train["hard"], sample_weight=train["weight"]) # 
        pred = clf.predict(valid[ineed[:-2]])
        cnf_matrix0 = confusion_matrix(valid["hard"], pred, labels=[0, 1], sample_weight=valid["weight"]) # 
        cnf_matrix += cnf_matrix0
    
    precision = cnf_matrix[1,1] / (cnf_matrix[0,1]+cnf_matrix[1,1])
    recall = cnf_matrix[1,1] / (cnf_matrix[1,0]+cnf_matrix[1,1])
    accuracy = (cnf_matrix[1,1] + cnf_matrix[0,0]) / cnf_matrix.sum()
    f_measure = 2 * precision * recall / (precision + recall)
    return [precision, recall, accuracy, f_measure]

# 手动调参
cv1 = cross_validation_SVC('1', 5, 1, 0.006)
cv2 = cross_validation_SVC('2', 5, 0.5, 0.01)
cv3 = cross_validation_SVC('3', 5, 0.7, 0.012)
cv4 = cross_validation_SVC('4', 5, 1.3, 0.009)
cv5 = cross_validation_SVC('5', 5, 30, 0.001)
cv_result = np.array([cv1, cv2, cv3, cv4, cv5])