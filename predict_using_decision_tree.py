from functions import *

pca5_df = pd.read_csv(workdir3+'5_pca_features.csv')
pca5_df['index'] = pca5_df['index'].astype(np.int)
del pca5_df['Unnamed: 0']

pca5_df["hard"] = 0
pca5_df.loc[pca5_df['m_vrtg'] >= 0.4, 'hard'] = 1

pca5_df["weight"] = 0
pca5_df.loc[pca5_df['hard'] == 0,"weight"] = w_light = pca5_df["hard"].sum() / float(pca5_df.shape[0])
pca5_df.loc[pca5_df['hard'] == 1,"weight"]= w_hard = 1 - w_light
class_weight = {0: w_light, 1:w_hard}

from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier

def cross_validation_DT(t, k, md, mss):
    kf = KFold(pca5_df.shape[0], n_folds=5)
    ineed = filter(lambda c: c[3] in t, pca5_df.columns) + ['hard', 'weight']
    cnf_matrix = np.zeros((2,2))
    cnf_matrix00 = np.zeros((2,2))
    for train_index, valid_index in kf:
        train = pca5_df.loc[train_index, ineed]
        valid = pca5_df.loc[valid_index, ineed]
        clf = DecisionTreeClassifier(random_state=0, max_depth=md, min_samples_split=mss, class_weight=class_weight)
        clf.fit(train[ineed[:-2]], train["hard"])
        pred = clf.predict(valid[ineed[:-2]])
        cnf_matrix0 = confusion_matrix(valid["hard"], pred, labels=[0, 1], sample_weight=valid["weight"])
        cnf_matrix += cnf_matrix0
        
#         cnf_matrix00 += confusion_matrix(valid["hard"], pred, labels=[0, 1]) 
#     ax = fig.add_subplot(111)
#     plot_confusion_matrix(cnf_matrix.round(2), classes=['light', 'hard'], title='{0} 10s'.format(t))
    
    precision = cnf_matrix[1,1] / (cnf_matrix[0,1]+cnf_matrix[1,1])
    recall = cnf_matrix[1,1] / (cnf_matrix[1,0]+cnf_matrix[1,1])
    accuracy = (cnf_matrix[1,1] + cnf_matrix[0,0]) / cnf_matrix.sum()
    f_measure = 2 * precision * recall / (precision + recall)
    return [precision, recall, accuracy, f_measure]

cv_dt_1 = cross_validation_DT('1', 10, 3, 2)
cv_dt_2 = cross_validation_DT('2', 10, 2, 2)
cv_dt_3 = cross_validation_DT('3', 10, 2, 3)
cv_dt_4 = cross_validation_DT('4', 10, 4, 2)
cv_dt_5 = cross_validation_DT('5', 10, 2, 2)
cv_dt_result = np.array([cv_dt_1, cv_dt_2, cv_dt_3, cv_dt_4, cv_dt_5])