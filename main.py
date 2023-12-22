import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics

df = pd.read_csv("/Users/developer/PycharmProjects/svm/Social_Network_Ads.csv")
few_rows = df.head()
shape = df.shape

X = df.iloc[:, [2, 3]]
y = df.iloc[:, 4]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=0);
# print("Training Data", X_train.shape)  # on the basis of 300 record we will train algorithm
# print("Training Data", X_test.shape)  # on the basic of training we will test 100 records

sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_train)
X_Test = sc_X.transform(X_test)


classifer = SVC(kernel="linear", random_state=0)

classifer.fit(X_Train, Y_train)
Y_predict = classifer.predict(X_Test)

print("Accuracy with Linear kernal is: ", metrics.accuracy_score(Y_test, Y_predict))

classifer = SVC(kernel="rbf")
classifer.fit(X_Train, Y_train)
Y_predict = classifer.predict(X_Test)
print("Accuracy with default RBF kernal is: ", metrics.accuracy_score(Y_test, Y_predict))

classifer = SVC(kernel="rbf", gamma=15, C=7, random_state=0)
classifer.fit(X_Train, Y_train)
Y_predict = classifer.predict(X_Test)
print("Accuracy with RBF kernal include gamma and C is : ", metrics.accuracy_score(Y_test, Y_predict))


# visaulizatio of Training Data
plt.scatter(X_Train[:, 0], X_Train[:, 1], c=Y_train)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('Training Data')
plt.show()


# drawing a hyperplane
classifer = SVC(kernel='linear', random_state=0)
classifer.fit(X_Train, Y_train)

# predicting test result
Y_predict = classifer.predict(X_Test)

# Plot data points
plt.scatter(X_Test[:, 0], X_Test[:, 1], c=Y_test)

# creating hyperplane
w = classifer.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2.5, 2.5)
yy = a * xx - (classifer.intercept_[0] / w[1])

# plot the hyperplane
plt.plot(xx, yy)
plt.axis('off')
plt.show()



