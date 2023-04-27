import pandas as pd
from numpy import genfromtxt, argmax
from sklearn import svm
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot
from tensorflow.keras.utils import plot_model


#Load the dataset
path = '1000_phone_records.csv'
df = pd.read_csv(path,header=0)

tdf = pd.read_csv('1000_phone_records.csv',header=0)

#Split into input and output columns
#There is a 9 because the shape needs to be 9
x,y = df.values[:, :9], df.values[:,-1]

#Ensure all data are floating point values
x = x.astype('int')

#Encode strings into integers
y = LabelEncoder().fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=0)
print("Train/Test data shape: ", x_train.shape, x_test.shape, y_train.shape, y_test.shape, "\n")

#Determine the number of input features
n_features = x_train.shape[1]

#Prints Statistics
precision_scores_list = []
accuracy_scores_list = []

def print_stats_metrics(y_test, y_pred):
    print('Accuracy: %.2f' % accuracy_score(y_test,   y_pred) )
    #Accuracy: 0.84
    accuracy_scores_list.append(accuracy_score(y_test,   y_pred) )
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print ("confusion matrix")
    print(confmat)
    print (pd.crosstab(y_test, y_pred, rownames=['flag'], colnames=['Predicted'], margins=True))
    precision_scores_list.append(precision_score(y_true=y_test, y_pred=y_pred,average='weighted'))
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred,average="weighted"))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred,average='weighted'))
    print('F1-measure: %.3f' % f1_score(y_true=y_test, y_pred=y_pred,average='weighted'))


#Support Vector Machine Algorithm
print("Support Vector Machine ")
clfSvm = svm.SVC(gamma = 'scale')
clfSvm.fit(x_train, y_train)
predictions = clfSvm.predict(x_test)
print_stats_metrics(y_test, predictions)

#Creates a new line
print("\n")

#Define the MLP model
model = Sequential()
model.add(Dense(10, activation='relu',kernel_initializer='he_normal',input_shape=(n_features,)))
model.add(Dense(8, activation='relu',kernel_initializer='he_normal'))
model.add(Dense(3, activation='softmax'))

#Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#Fit the model; try verbose =1
history = model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=1, shuffle=True)

#Evaluate the model
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)
print('Loss: %.3f' % loss)

#Show model
model.summary()
plot_model(model, 'model.png', show_shapes=True)

#Plot learning curves
pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Cross Entropy')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='val')
pyplot.legend()
pyplot.show()

#Make a prediction
row = [5663958440,0,0,0,0,0,0,0,0]
yhat = model.predict(tdf)
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
