import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import aif360
from aif360.metrics import ClassificationMetric
from common_utils import compute_metrics

np.random.seed(1)

def bias_metrics_lr(privileged_groups, unprivileged_groups):
    # train test split
    dataset_orig_train, dataset_orig_vt = binaryLabelDataset.split([0.7], shuffle=True)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)

    # Logistic regression classifier and predictions
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()
    w_train = dataset_orig_train.instance_weights.ravel()

    lmod = LogisticRegression(solver='lbfgs', max_iter=1000)
    lmod.fit(X_train, y_train, 
            sample_weight=dataset_orig_train.instance_weights)
    y_train_pred = lmod.predict(X_train)

    # positive class index
    pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]

    dataset_orig_train_pred = dataset_orig_train.copy()
    dataset_orig_train_pred.labels = y_train_pred

    dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
    y_valid = dataset_orig_valid_pred.labels
    dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)

    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale_orig.transform(dataset_orig_test_pred.features)
    y_test = dataset_orig_test_pred.labels
    dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)

    num_thresh = 100
    ba_arr = np.zeros(num_thresh)
    class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
    for idx, class_thresh in enumerate(class_thresh_arr):
        
        fav_inds = dataset_orig_valid_pred.scores > class_thresh
        dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
        dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label
        
        classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                                dataset_orig_valid_pred, 
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)
        
        ba_arr[idx] = 0.5*(classified_metric_orig_valid.true_positive_rate()\
                        +classified_metric_orig_valid.true_negative_rate())

    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_class_thresh = class_thresh_arr[best_ind]

    print("Best balanced accuracy (no reweighing) = %.4f" % np.max(ba_arr))
    print("Optimal classification threshold (no reweighing) = %.4f" % best_class_thresh)

    bal_acc_arr_orig = []
    disp_imp_arr_orig = []
    avg_odds_diff_arr_orig = []

    print("\nClassification threshold used = %.4f" % best_class_thresh)
    for thresh in tqdm(class_thresh_arr):
        
        if thresh == best_class_thresh:
            disp = True
        else:
            disp = False
        
        fav_inds = dataset_orig_test_pred.scores > thresh
        dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
        dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label
        
        metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred, 
                                        unprivileged_groups, privileged_groups,
                                        disp = disp)

        bal_acc_arr_orig.append(metric_test_bef["Balanced accuracy"])
        avg_odds_diff_arr_orig.append(metric_test_bef["Average odds difference"])
        disp_imp_arr_orig.append(metric_test_bef["Disparate impact"])
    
def bias_metrics_rf(privileged_groups, unprivileged_groups):
    # train test split
    dataset_orig_train, dataset_orig_vt = binaryLabelDataset.split([0.7], shuffle=True)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)

    # Random Forest classifier and predictions
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()
    w_train = dataset_orig_train.instance_weights.ravel()

    rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
    rf.fit(X_train, y_train);
    y_train_pred = rf.predict(X_train)

    # positive class index
    pos_ind = np.where(rf.classes_ == dataset_orig_train.favorable_label)[0][0]

    dataset_orig_train_pred = dataset_orig_train.copy()
    dataset_orig_train_pred.labels = y_train_pred

    dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
    y_valid = dataset_orig_valid_pred.labels
    dataset_orig_valid_pred.scores = rf.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)

    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale_orig.transform(dataset_orig_test_pred.features)
    y_test = dataset_orig_test_pred.labels
    dataset_orig_test_pred.scores = rf.predict_proba(X_test)[:,pos_ind].reshape(-1,1)

    num_thresh = 100
    ba_arr = np.zeros(num_thresh)
    class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
    for idx, class_thresh in enumerate(class_thresh_arr):
        
        fav_inds = dataset_orig_valid_pred.scores > class_thresh
        dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
        dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label
        
        classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                                dataset_orig_valid_pred, 
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)
        
        ba_arr[idx] = 0.5*(classified_metric_orig_valid.true_positive_rate()\
                        +classified_metric_orig_valid.true_negative_rate())

    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_class_thresh = class_thresh_arr[best_ind]

    print("Best balanced accuracy (no reweighing) = %.4f" % np.max(ba_arr))
    print("Optimal classification threshold (no reweighing) = %.4f" % best_class_thresh)

    bal_acc_arr_orig = []
    disp_imp_arr_orig = []
    avg_odds_diff_arr_orig = []

    print("\nClassification threshold used = %.4f" % best_class_thresh)
    for thresh in tqdm(class_thresh_arr):
        
        if thresh == best_class_thresh:
            disp = True
        else:
            disp = False
        
        fav_inds = dataset_orig_test_pred.scores > thresh
        dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
        dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label
        
        metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred, 
                                        unprivileged_groups, privileged_groups,
                                        disp = disp)

        bal_acc_arr_orig.append(metric_test_bef["Balanced accuracy"])
        avg_odds_diff_arr_orig.append(metric_test_bef["Average odds difference"])
        disp_imp_arr_orig.append(metric_test_bef["Disparate impact"])

# ----------------------------------
#            READ DATA
# ----------------------------------

# read data
df = pd.read_csv('data/all_census_aif360_data.csv')

# one hot encode helpful columns
categoricalFeatures = ['Poverty_Rate', 'Education_Rate', 'Employment_Rate', 'PREDICTOR RAT AGE AT LATEST ARREST', 'PREDICTOR RAT VICTIM SHOOTING INCIDENTS', 'PREDICTOR RAT VICTIM BATTERY OR ASSAULT', 'PREDICTOR RAT ARRESTS VIOLENT OFFENSES', 'PREDICTOR RAT GANG AFFILIATION', 'PREDICTOR RAT NARCOTIC ARRESTS', 'PREDICTOR RAT TREND IN CRIMINAL ACTIVITY', 'PREDICTOR RAT UUW ARRESTS']

for feature in categoricalFeatures:
    onehot = pd.get_dummies(df[feature], prefix=feature)
    df = df.drop(feature, axis=1)
    df = df.join(onehot)

# ----------------------------------
#        FAIRNESS FOR RACE
# ----------------------------------

print('\n\n--------------------------------\n LOGISTIC REGRESSION RACE vs SSL SCORE BIAS METRICS\n--------------------------------')

binaryLabelDataset = aif360.datasets.BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=df,
    label_names=['SSL SCORE'],
    protected_attribute_names=['RACE CODE CD'])

#dividing the dataset into train and test
dataset_orig_train, dataset_orig_test = binaryLabelDataset.split([0.7], shuffle=True)

# Priviliged group: White (1)
# Unpriviliged group: Non-White (0)
privileged_groups = [{'RACE CODE CD': 1}]
unprivileged_groups = [{'RACE CODE CD': 0}]

bias_metrics_lr(privileged_groups, unprivileged_groups)

print('\n\n--------------------------------\n RANDOM FOREST RACE vs SSL SCORE BIAS METRICS\n--------------------------------')

bias_metrics_rf(privileged_groups, unprivileged_groups)


# ----------------------------------
#         FAIRNESS FOR SEX
# ----------------------------------

print('\n\n--------------------------------\n LOGISTIC REGRESSION SEX vs SSL SCORE BIAS METRICS\n--------------------------------')

binaryLabelDataset = aif360.datasets.BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=df,
    label_names=['SSL SCORE'],
    protected_attribute_names=['SEX CODE CD'])

#dividing the dataset into train and test
dataset_orig_train, dataset_orig_test = binaryLabelDataset.split([0.7], shuffle=True)

# Priviliged group: Male (1)
# Unpriviliged group: Female (0)
privileged_groups = [{'SEX CODE CD': 1}]
unprivileged_groups = [{'SEX CODE CD': 0}]

bias_metrics_lr(privileged_groups, unprivileged_groups)

print('\n\n--------------------------------\n RANDOM FOREST SEX vs SSL SCORE BIAS METRICS\n--------------------------------')

bias_metrics_rf(privileged_groups, unprivileged_groups)

# ----------------------------------
#         FAIRNESS FOR AGE
# ----------------------------------

print('\n\n--------------------------------\n LOGISTIC REGRESSION AGE vs SSL SCORE BIAS METRICS\n--------------------------------')

binaryLabelDataset = aif360.datasets.BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=df,
    label_names=['SSL SCORE'],
    protected_attribute_names=['AGE GROUP'])

#dividing the dataset into train and test
dataset_orig_train, dataset_orig_test = binaryLabelDataset.split([0.7], shuffle=True)

# Priviliged group: Old People (Over 30) (1)
# Unpriviliged group: Young People (20-30) (0)
privileged_groups = [{'AGE GROUP': 1}]
unprivileged_groups = [{'AGE GROUP': 0}]

bias_metrics_lr(privileged_groups, unprivileged_groups)

print('\n\n--------------------------------\n RANDOM FOREST AGE vs SSL SCORE BIAS METRICS\n--------------------------------')

bias_metrics_rf(privileged_groups, unprivileged_groups)