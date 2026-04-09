import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_collection import fetch_dataset
from normalization import *
from knn import KNN
from evaluation import cross_validation,evaluate


df=fetch_dataset()

data=df.values

np.random.seed(42)
np.random.shuffle(data)

split=int(0.8*len(data))

train=data[:split]
test=data[split:]

X_train=train[:,:-1]
y_train=train[:,-1]

X_test=test[:,:-1]
y_test=test[:,-1]


scalers={

"Raw":None,
"MinMax":MinMaxScaler(),
"ZScore":ZScoreScaler(),
"Robust":RobustScaler(),
"MeanNorm":MeanNormalizer(),
"MaxAbs":MaxAbsScaler(),
"Decimal":DecimalScaling(),
"ModifiedZ":ModifiedZScore(),
"Pareto":ParetoScaler(),
"Log":LogTransform(),
"Reciprocal":ReciprocalTransform(),
"Sqrt":SqrtTransform(),
"BoxCox":BoxCox(),
"YeoJohnson":YeoJohnson(),
"Tanh":TanhScaler(),
"Sigmoid":SigmoidScaler(),
"L1Norm":L1Normalizer(),
"L2Norm":L2Normalizer(),
"Softmax":SoftmaxScaler(),
"Quantile":QuantileNormalizer(),
"RankGauss":RankGauss()

}


results=[]


for name,scaler in scalers.items():

    if scaler is None:

        Xtr=X_train
        Xte=X_test

    else:

        scaler.fit(X_train)

        Xtr=scaler.transform(X_train)
        Xte=scaler.transform(X_test)

    best_k=cross_validation(Xtr,y_train)

    model=KNN(best_k)

    model.fit(Xtr,y_train)

    acc,f1=evaluate(model,Xte,y_test)

    results.append([name,best_k,acc,f1])


results_df=pd.DataFrame(results,
columns=["Method","Best_k","Accuracy","Macro_F1"])

results_df=results_df.sort_values(by="Accuracy",ascending=False)

results_df["Rank"]=range(1,len(results_df)+1)

print(results_df)

results_df.to_csv("results_table.csv",index=False)


best_method=results_df.iloc[0]["Method"]
worst_method=results_df.iloc[-1]["Method"]

methods=["Raw",best_method,worst_method]


plt.figure(figsize=(12,4))

for i,method in enumerate(methods):

    scaler=scalers[method]

    if scaler is None:
        X=X_train
    else:
        scaler.fit(X_train)
        X=scaler.transform(X_train)

    plt.subplot(1,3,i+1)
    sns.histplot(X[:,0],kde=True)
    plt.title(method)

plt.tight_layout()
plt.savefig("feature_distribution.png")
plt.show()


plt.figure(figsize=(7,4))

for method in methods:

    scaler=scalers[method]

    if scaler is None:
        X=X_train
    else:
        scaler.fit(X_train)
        X=scaler.transform(X_train)

    scores=[]

    for k in range(1,31):

        model=KNN(k)
        model.fit(X,y_train)

        pred=model.predict(X_test)

        scores.append(np.mean(pred==y_test))

    plt.plot(range(1,31),scores,label=method)

plt.xlabel("k")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("accuracy_vs_k.png")
plt.show()
