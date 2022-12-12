import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC

iris = sns.load_dataset('iris')
iris.head()

sns.pairplot(iris, hue='species', palette='Dark2')

x = iris.drop('species', axis=1)
y = iris['species']
X_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# from support vector machine algorithm
svm = SVC()
svm.fit(X_train, y_train)
pred = svm.predict(x_test)
print(classification_report(y_test, pred))
print('/n')
print(confusion_matrix(y_test, pred))

# from grid_search algorithm
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose=True)
grid.fit(X_train, y_train)
pred_g = grid.predict(x_test)
print(classification_report(y_test, pred_g))
print('/n')
print(confusion_matrix(y_test, pred_g))
