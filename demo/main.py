from sklearn import tree

#[height, weight, shoe size]

X = [[181,80,44], [190,90,45], [165,50,38], [175,73,42]]

Y = ['male','male','famale','famale']

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X,Y)

prediction = clf.predict([[179,77,40]])

print(str(prediction))