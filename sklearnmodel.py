from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import loadimage as lm

Dataset_path = 'F:\Masters RIME\Third Semester\Deep learning\Assignemnts\Train_Images'
Prediction_files_path = 'F:\Masters RIME\Third Semester\Deep learning\Assignments\Test_Images'

x_train,y_train = lm.train(Dataset_path)
x_test,y_test = lm.test(Prediction_files_path)

classifier = MLPClassifier(hidden_layer_sizes=(8,7,6), max_iter=1000,activation = 'relu',solver='adam',random_state=1,alpha=0.00095)
classifier.fit(x_train, y_train)

y_pred_train = classifier.predict(x_train)
y_pred_test = classifier.predict(x_test)

print('accuracy for training set is: '+ str(accuracy_score(y_train,y_pred_train)))
print('accuracy for test set is: '+ str(accuracy_score(y_test,y_pred_test)))
