import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
import numpy as np
import pickle


def save_model(model, model_name):
    # save the model to disk
    filename = 'models/' + model_name + '.sav'
    pickle.dump(model, open(filename, 'wb'))
    print("Model Saved Successfully")


def load_model(model_name):
    # load the model from disk
    filename = 'models/' + model_name + '.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

# This Module will have the models that we will use to train the dataset
# 1- SVM
# 3- KNN
# 4- Logistic Regression
# 5- Decision Tree
# 6- Naive Bayes
# 7- Neural Network


# SVM


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
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)
    # Fit the grid search to the data
    grid_search.fit(train_features, train_labels)
    best_grid = grid_search.best_estimator_
    # Predict the labels of the test set: y_pred
    y_pred = best_grid.predict(test_features)
    # Compute and print metrics
    print("Accuracy: \n{}".format(accuracy_score(test_labels, y_pred)))
    print("Confusion Matrix: \n{}".format(confusion_matrix(test_labels, y_pred)))
    print("Classification Report: \n{}".format(classification_report(test_labels, y_pred)))
    return best_grid


def nn_model(train_features, train_labels, test_features, test_labels , epochs=10 , num = 0):
    # Adding the input layer and the first hidden layer
    input_size = train_features.shape[1]
    inp = tf.keras.Input(shape=(input_size,))
    x1 = tf.keras.layers.Dense(input_size, activation=tf.nn.leaky_relu)(inp)
    # Adding the second hidden layer
    x2 = tf.keras.layers.Dense(input_size//2, activation=tf.nn.leaky_relu)(x1)
    # Adding the output layer
    output = tf.keras.layers.Dense(6, activation=tf.nn.softmax)(x2)
    classifier = tf.keras.Model(inputs=inp, outputs=output)
    # Compiling the ANN
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min' , patience=3 , verbose=1)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('models/nn_model'+str(num)+'.h5', save_best_only=True)
    classifier.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Fitting the ANN to the Training set
    classifier.fit(train_features, train_labels, batch_size=16,
                   epochs=epochs, validation_data=(test_features, test_labels)  , callbacks=[early_stopping , model_checkpoint])
    
    # Predicting the Test set results
    y_pred = classifier.predict(test_features)
    y_pred = np.argmax(y_pred, axis=1)
    # Compute and print metrics 
    print("Accuracy: \n{}".format(accuracy_score(test_labels, y_pred)))
    print("Confusion Matrix: \n{}".format(confusion_matrix(test_labels, y_pred)))
    print("Classification Report: \n{}".format(classification_report(test_labels, y_pred)))
    return classifier
