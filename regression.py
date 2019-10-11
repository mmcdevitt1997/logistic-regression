import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('titanic_train.csv')
test = pd.read_csv('titanic_test.csv')

print(train.head())

# sns.countplot(x='SibSp', data=train)

# train['Fare'].hist(bins=40)

# train['Fare'].plot(kind='hist', bins=30)

sns.boxplot(x='Pclass', y='Age', data=train)
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)

sns.heatmap(train.isnull(), yticklabels=False, cbar=False)


train.drop('Cabin', axis=1, inplace=True)
train.drop (['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

X = train.drop('Survived', axis=1 )
y= train['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.30, random_state=101)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))