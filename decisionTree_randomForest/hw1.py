import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.tree import export_graphviz
import graphviz


df = pd.read_csv('titanic.csv')

#Creating a Survived column with text
str_survived = []
for row in df['Survived']:
    if row == 0 :  str_survived.append('Dead')
    else :  str_survived.append('Alive')

df['Survival_state'] = str_survived


#Creating X and Y values
X = df.iloc[:, 0:2]
y = df.loc[:,['Survived']]


#Splitting the data to 1/4 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = None)


#Creating the model
model = tree.DecisionTreeClassifier(max_depth = 5)
model.fit(X,y)


#Visualization of the model
dot_data = export_graphviz(
            model,
            out_file =  None,
            feature_names = list(X.columns),
            class_names = df['Survival_state'].unique(),
            filled = True,
            rounded = True)

graph = graphviz.Source(dot_data)
graph.render(filename = 'titanic', format = 'png')


#Creating a prediction of the test results
y_pred = model.predict(X_test)

#Make confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
plt.show()

#Calculate accuracy, precision, and recall score
acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
print (f'DT Acc score: {acc:.2f} ')
print (f'Recall score: {recall:.2f} ')
print (f'Precision score: {precision:.2f} ')



#Make two test passengers
new_passengers = [{'PClass':1, 'Age':17},
               {'PClass':3, 'Age':17}]
new_data = pd.DataFrame(new_passengers)


#Predict with new data and create dataframe
new_y = pd.DataFrame(model.predict(new_data))

# apply species information based on the prediction
new_y[1] = new_y[0].apply(lambda x: 'Dead' if x == 0  else ('Alive'))




