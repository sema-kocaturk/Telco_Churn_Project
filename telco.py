import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv(r"C:\Users\user\Desktop\Datas\Telco-Customer-Churn.csv")

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

## Keşifçi Veri Analizi
#Adım 1: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car_cols: list
                Kategorik görünümlü kardinal değişken listesi
        num_but_cat_cols: list
                Numerik görünümlü kategorik değişken listesi


    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Note
    ------
        cat_cols + num_cols + cat_but_car_cols = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car_cols = değişken
        sayısı

    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_but_cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and
                        dataframe[col].nunique() < cat_th]
    cat_but_car_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and
                        dataframe[col].nunique() > car_th]

    cat_cols = cat_cols + num_but_cat_cols
    cat_cols = [col for col in cat_cols if col not in cat_but_car_cols]
    # num_cols
    num_cols = [col for col in num_cols if col not in num_but_cat_cols]

    print("cat_cols:", cat_cols)
    print("num_cols:", num_cols)
    print("num_but_cat_cols:", num_but_cat_cols)
    print("cat_but_car_cols:", cat_but_car_cols)
    return cat_cols, num_cols, num_but_cat_cols, cat_but_car_cols

cat_cols, num_cols,num_but_cat_cols,cat_but_car_cols = grab_col_names(df)


df.info()

#Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

def change_type(dataframe):
    for col in dataframe:
        if col in num_but_cat_cols:
          dataframe[col] = dataframe[col].astype(object)


change_type(df)
le = LabelEncoder()
for col in cat_cols:
   df["Churn"] = le.fit_transform(df["Churn"])

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.isnull().sum()
cat_cols, num_cols,num_but_cat_cols,cat_but_car_cols = grab_col_names(df)
df.info()

#Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

#Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)

#Adım 5: Aykırı gözlem var mı inceleyiniz.
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))

#Adım 6: Eksik gözlem var mı inceleyiniz.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

df.isnull().sum()


## Görev 2 : Feature Engineering

# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.


for col in num_cols:
    df[col] = df[col].fillna(df.groupby(cat_cols)[col].transform("mean"))

df.dropna(inplace=True)

#Adım 2: Yeni değişkenler oluşturunuz.
for col in df.columns:
    print(col, ":", df[col].nunique())
df.describe()
df.info()
df["tenure"].describe()
df.head()
df["Term_Cat"] = pd.cut(df["tenure"], bins=[0, 10, 30, 55,75], labels=["New", "Mid", "Old","Seniour"])


df["MonthlyCharges"].describe()
df["Charge_Cat"] = pd.cut(df["MonthlyCharges"], bins=[0, 35, 70,90,120],
                                  labels=["Lower", "Standart", "High", "Extreme"])


cat_cols, num_cols,num_but_cat_cols,cat_but_car_cols = grab_col_names(df)

#Adım 3: Encoding işlemlerini gerçekleştiriniz.

# Label Encoding
binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
               and df[col].nunique() == 2]

df[binary_cols].head()


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


for col in binary_cols:
    label_encoder(df, col)

df[binary_cols].head()
df.head()

# One Hot Encoding
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, ohe_cols)
df.head()

#Adım 4: Numerik değişkenler için standartlaştırma yapınız.
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


df[num_cols].head()


#Görev 3 : Modelleme
#Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz
y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)

#Logistic Regression
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(max_iter=500).fit(X, y)
log_model.intercept_
log_model.coef_
y_pred = log_model.predict(X)
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

# ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)



log_cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])
log_accuracy = log_cv_results["test_accuracy"].mean()
log_f1 = log_cv_results["test_f1"].mean()
log_roc_auc= log_cv_results["test_roc_auc"].mean()




#Random Forest
rf_model= RandomForestClassifier()
rf_cv_results= cross_validate(rf_model,
                              X,y,
                              cv= 5,
                              scoring=["f1","accuracy","roc_auc"])
rf_accuracy = rf_cv_results['test_accuracy'].mean()
rf_f1 = rf_cv_results['test_f1'].mean()
rf_roc_auc = rf_cv_results['test_roc_auc'].mean()

#GBM
gbm_model = GradientBoostingClassifier()
gbm_cv_results = cross_validate(gbm_model,
                                X,y,
                                cv=5,
                                scoring=["f1","accuracy","roc_auc"])
gbm_accuracy = gbm_cv_results['test_accuracy'].mean()
gbm_f1 = gbm_cv_results['test_f1'].mean()
gbm_roc_auc = gbm_cv_results['test_roc_auc'].mean()


#Xgboost

Xgboost_model = XGBClassifier()
xgboost_cv_results= cross_validate(Xgboost_model,
                                   X,y,
                                   cv=5,
                                   scoring=["f1","accuracy","roc_auc"])

xgboost_accuracy = xgboost_cv_results['test_accuracy'].mean()
xgboost_f1 = xgboost_cv_results['test_f1'].mean()
xgboost_roc_auc = xgboost_cv_results['test_roc_auc'].mean()


#LightGBM

lgbm_model = LGBMClassifier()
lgbm_cv_results = cross_validate(lgbm_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])
lgbm_accuracy = lgbm_cv_results['test_accuracy'].mean()
lgbm_f1 = lgbm_cv_results['test_f1'].mean()
lgbm_roc_auc = lgbm_cv_results['test_roc_auc'].mean()

#catboost

catboost_model = CatBoostClassifier()
catboost_cv_results = cross_validate(catboost_model,
                                     X,y,
                                     cv = 5,
                                    scoring = ["f1","accuracy","roc_auc"])

catboost_f1 = catboost_cv_results['test_f1'].mean()
catboost_accuracy = catboost_cv_results['test_accuracy'].mean()
catboost_roc_auc = catboost_cv_results['test_roc_auc'].mean()

#Decision Tree
dt_model = DecisionTreeClassifier().fit(X, y)
dt_cv_results = cross_validate(dt_model,
                               X, y,
                               cv=5,
                               scoring=["accuracy", "f1", "roc_auc"])
dt_test = dt_cv_results['test_accuracy'].mean()
dt_f1 = dt_cv_results['test_f1'].mean()
dt_auc = dt_cv_results['test_roc_auc'].mean()

best_model_results = pd.DataFrame(
    {"Model": ["Logistic Regression", "Random Forest", "GBM", "LightGBM", "XGBoost",  "Decision Tree"],
     "Accuracy": [log_accuracy, rf_accuracy, gbm_accuracy, lgbm_accuracy, xgboost_accuracy, dt_test],
     "AUC": [log_roc_auc, rf_roc_auc, gbm_roc_auc, lgbm_roc_auc, xgboost_accuracy, dt_auc],
     "F1_Score": [log_f1, rf_f1, gbm_f1, lgbm_f1, xgboost_f1,  dt_f1]})

best_model_results = best_model_results.sort_values("Accuracy", ascending = False)

# Selected Models: GBM, Logistic Regression, LightGBM, Random Forest

#GBM
gbm_model=GradientBoostingClassifier()
gbm_model.get_params()
gbm_params = {"learning_rate": [0.01, 0.1,0.2],
              "max_depth": [2,3, 8, 10],
              "n_estimators": [50,100, 500, 1000],
              "subsample": [1, 0.5, 2]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)
gbm_cv_results = cross_validate(gbm_final,
                                X,y,
                                cv=5,
                                scoring=["f1","accuracy","roc_auc"])
gbm_final_accuracy = gbm_cv_results['test_accuracy'].mean()
gbm_final_f1 = gbm_cv_results['test_f1'].mean()
gbm_final_roc_auc = gbm_cv_results['test_roc_auc'].mean()


#Logistic Regression
log_model = LogisticRegression(max_iter=500)
log_model.get_params()
log_params = {"penalty": ['l1', 'l2'],
              'C': np.logspace(-3, 3, 7),
              "solver": ['newton-cg', 'lbfgs', 'liblinear']}
log_best_grid = GridSearchCV(log_model, log_params,cv=5,n_jobs=-1,verbose=True).fit(X,y)
log_final = log_model.set_params(**log_best_grid.best_params_).fit(X,y)
log_cv_results=cross_validate(log_final,
                              X,y,
                              cv=5,
                              scoring=["f1","accuracy","roc_auc"])
log_final_accuracy = log_cv_results['test_accuracy'].mean()
log_final_f1 = log_cv_results['test_f1'].mean()
log_final_roc_auc = log_cv_results['test_roc_auc'].mean()

#Light GBM

lgbm_model.get_params()
lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}
lgbm_best_grid = GridSearchCV(lgbm_model,log_params,cv=5,n_jobs=-1,verbose=True).fit(X,y)
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_).fit(X,y)
lgbm_cv_results = cross_validate(lgbm_final,
                               X,y,
                               cv=5,
                               scoring=["f1","accuracy","roc_auc"])

lgbm_final_accuracy = lgbm_cv_results['test_accuracy'].mean()
lgbm_final_f1 = lgbm_cv_results['test_f1'].mean()
lgbm_final_roc_auc = lgbm_cv_results['test_roc_auc'].mean()

#Random Forest

rf_model= RandomForestClassifier()
rf_model.get_params()
rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}
rf_best_grid = GridSearchCV(rf_model,rf_params,cv=5,n_jobs=-1,verbose=True).fit(X,y)
rf_final = rf_model.set_params(**rf_best_grid.best_params).fit(X,y)
rf_cv_results = cross_validate(rf_final,
                               X,y,
                               cv=5,
                               scoring=["f1","accuracy","roc_auc"])

rf_final_accuracy = rf_cv_results['test_accuracy'].mean()
rf_final_f1 = rf_cv_results['test_f1'].mean()
rf_final_roc_auc = rf_cv_results['test_roc_auc'].mean()




def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)
plot_importance(gbm_final, X)
plot_importance(lgbm_final, X)
plot_importance(log_final, X)