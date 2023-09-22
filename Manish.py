from turtle import color
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
customer_churn = pd.read_csv(r'WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(customer_churn.head())
c_5 = customer_churn.iloc[:, 4]
print(c_5.head())

c_random = customer_churn[(customer_churn["gender"] == "Male") & (customer_churn["SeniorCitizen"] == 1) & (customer_churn["PaymentMethod"] == "Electronic check")]
print(c_random.head())

c_random = customer_churn[(customer_churn["tenure"] > 70) | (customer_churn["MonthlyCharges"] > 100)]
print(c_random.head())

c_random = customer_churn[(customer_churn["Contract"] == "Two Year") | (customer_churn["PaymentMethod"] == 'Mailed check') & (customer_churn["Churn"] == "Yes")]
print(c_random.head())

c_333 = customer_churn.sample(333)
print(c_333.head())
customer_churn['Churn'].value_counts()

bote = customer_churn["InternetService"].value_counts().keys().tolist()
prince = customer_churn["InternetService"].value_counts().tolist()
plt.bar(bote, prince, color="brown")
plt.xlabel("Categories of Internet")
plt.ylabel("Count")
plt.title("Distribution of internet")
plt.show()

plt.hist(customer_churn['tenure'], bins=30, color='yellow')
plt.title("Distribution of tenure")
plt.show()

x = customer_churn['tenure']
y = customer_churn['MonthlyCharges']
plt.scatter(x, y)
plt.xlabel('tenure')
plt.ylabel('MonthlyCharges')
plt.title("MonthlyCharges vs tenure")
plt.show()

column = ['tenure']
by = ['Contract']
customer_churn.boxplot(column, by)
plt.show()

from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x = customer_churn[['tenure']]
y = customer_churn[['MonthlyCharges']]
print(x.head())
print(y.head())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print(np.mean(x_train))
print(np.mean(x_test))
print(np.mean(y_train))
print(np.mean(y_test))
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test, y_pred))

x = customer_churn[["MonthlyCharges", "tenure"]]
y = customer_churn[['Churn']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(x_train, y_train)
y_pred = log_model.predict(x_test)
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_test, y_pred), accuracy_score(y_test, y_pred)

x = customer_churn[['tenure']]
y = customer_churn[['Churn']]
from sklearn.tree import DecisionTreeClassifier
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
my_tree = DecisionTreeClassifier()
my_tree.fit(x_train, y_train)
y_pred = my_tree.predict(x_test)
confusion_matrix(y_test, y_pred)
