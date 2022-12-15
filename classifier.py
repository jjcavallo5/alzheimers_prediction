'''
https://www.kaggle.com/datasets/brsdincer/alzheimer-features/versions/1?resource=download
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from data import load_data, standardize, visualize, save_conf_matrix


x_train, y_train, x_test, y_test = load_data()
x_train, x_test = standardize(x_train, x_test)

KNN     = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)
LogReg  = LogisticRegression(max_iter=1000).fit(x_train, y_train)
Tree    = DecisionTreeClassifier().fit(x_train, y_train)
NB      = GaussianNB().fit(x_train, y_train)
Net     = MLPClassifier((32), max_iter=2000).fit(x_train, y_train)

pred_KNN    = KNN.predict(x_test)
pred_LogReg = LogReg.predict(x_test)
pred_Tree   = Tree.predict(x_test)
pred_NB     = NB.predict(x_test)
pred_Net    = Net.predict(x_test)

print(sum(pred_KNN==y_test) / len(y_test))
print(sum(pred_LogReg==y_test) / len(y_test))
print(sum(pred_Tree==y_test) / len(y_test))
print(sum(pred_NB==y_test) / len(y_test))
print(sum(pred_Net==y_test) / len(y_test))

save_conf_matrix(KNN, y_test, pred_KNN, 'KNN')
save_conf_matrix(LogReg, y_test, pred_LogReg, 'LogReg')
save_conf_matrix(Tree, y_test, pred_Tree, 'Tree')
save_conf_matrix(NB, y_test, pred_NB, 'NB')
save_conf_matrix(Net, y_test, pred_Net, 'Net')

