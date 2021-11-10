import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# 視覺化參考：https://www.kaggle.com/aminjpr/iris-comprehensive-visualization-and-modeling

# Read Data
train = pd.read_csv("Iris.csv")
print(train.info())
print("_"*50)
print(train.columns)
# 共有150筆沒有遺失值的資料

# 顯示物種,各50筆的資料
print(train.groupby("Species").size())
print("確認資料型態正確，沒有遺失值，可以開始進行資料視覺化")

### Plotting countplot for each Species
sns.countplot(data=train, x='Species')
plt.show()

sns.heatmap(train.corr(),
            annot=True,
            cmap='RdYlBu')
plt.show()

### Plotting feature distribution
X = train.drop('Species', axis=1)
y = train.Species
for col in X.columns:
    fig, ax = plt.subplots(nrows=1, ncols=3,
                           sharey=True,
                           figsize=(12, 4))

    sns.boxplot(data=train, x='Species', y=col, ax=ax[0])
    sns.violinplot(data=train, x='Species', y=col, ax=ax[1])
    sns.histplot(data=train, y=col, ax=ax[2])

    plt.tight_layout()
    plt.show()

sns.pairplot(train, hue='Species', kind='reg')
plt.show()

g=sns.jointplot(data=train,
                x='PetalLengthCm', y='PetalWidthCm',
                hue='Species')

g.plot_joint(sns.kdeplot, color="r")
plt.show()

fig = px.scatter_3d(train,
                    x='SepalLengthCm', y='SepalWidthCm', z='PetalWidthCm',
                    color='Species')
fig.show()

### Plotting Radar plot
df_gb=train.groupby('Species').median().transpose().reset_index()

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
      r=df_gb['Iris-setosa'],
      theta=df_gb['index'],
      fill='toself',
      name='Iris-setosa'
))
fig.add_trace(go.Scatterpolar(
      r=df_gb['Iris-versicolor'],
      theta=df_gb['index'],
      fill='toself',
      name='Iris-versicolor'
))
fig.add_trace(go.Scatterpolar(
      r=df_gb['Iris-virginica'],
      theta=df_gb['index'],
      fill='toself',
      name='Iris-virginica'
))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 7]
    )),
  showlegend=False
)

fig.show()

### Plotting Andrews curve, visualizing multivariate data by mapping into 2D space through Fourier series
pd.plotting.andrews_curves(train, 'Species', colormap='viridis')
plt.legend(loc='upper left')

### Plotting a heatmap for sorted dataframe based on Species
sns.heatmap(train.sort_values('Species').drop('Species',axis=1))


# Modeling
### Splitting dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

print(X_train.shape, X_test.shape)

### Using a single Decision Tree for this classification
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(metrics.classification_report(y_pred, y_test))
metrics.plot_confusion_matrix(model, X_test, y_pred);

### Tree representation
plt.figure(figsize=(5,5), dpi=200)
tree.plot_tree(model,
               feature_names=train.columns,
               class_names=train.Species.unique(),
               filled=True)

plt.show()

### Text representation

txt = tree.export_text(model)
print(txt)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)
print(acc_log)
print(metrics.classification_report(y_pred, y_test))
metrics.plot_confusion_matrix(model, X_test, y_pred);


if __name__ == "__main__":
    print("啟動")