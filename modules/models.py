def save_model(model, model_name):
    # save the model to disk
    filename = 'models/' + model_name + '.sav'
    pickle.dump(model, open(filename, 'wb'))

def load_model(model_name):
    # load the model from disk
    filename = 'models/' + model_name + '.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

#This Module will have the models that we will use to train the dataset
#1- SVM
#2- Random Forest
#3- KNN
#4- Logistic Regression
#5- Decision Tree
#6- Naive Bayes
#7- Neural Network

#SVM
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def svm_model(train_features, train_labels, test_features, test_labels):
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf']
    }
    # Create a based model
    svm_model = svm.SVC()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = svm_model, param_grid = param_grid, 
                              cv = 3, n_jobs = -1, verbose = 2)
    # Fit the grid search to the data
    grid_search.fit(train_features, train_labels)
    best_grid = grid_search.best_estimator_
    # Predict the labels of the test set: y_pred
    y_pred = best_grid.predict(test_features)
    # Compute and print metrics
    print("Accuracy: {}".format(accuracy_score(test_labels, y_pred)))
    print("Confusion Matrix: {}".format(confusion_matrix(test_labels, y_pred)))
    print("Classification Report: {}".format(classification_report(test_labels, y_pred)))
    return best_grid