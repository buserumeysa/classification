import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('bkii.csv')
data.columns
data['BMI'] = data['Kilo'] / ((data['Boy'] / 100) ** 2)
X_train, X_test, y_train, y_test = train_test_split(data[['BMI']], data['Durum'], test_size=0.25, random_state=42)
classifier = DecisionTreeClassifier(max_depth=6)
classifier.fit(X_train, y_train)

# Tahminlerin yapılması -
y_pred = classifier.predict(X_test)

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, plot_confusion_matrix
print("Test accuracy: ", classifier.score(X_test, y_test))
print("Train accuracy: ",classifier.score(X_train, y_train))
print(confusion_matrix(y_test, y_pred)) #Zayif - 0, Normal - 1 ,Kilolu - 2,Obez -3 ,Morbid Obez -4.
print(classification_report(y_test, y_pred))
plot_confusion_matrix(classifier, X_test, y_test, display_labels=["Zayif"," Normal","Kilolu" ,"Obez" ,"Morbid Obez"])
# Karar ağacı ve ağaç modülü
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=6)
classifier.fit(X_train, y_train)

# Tahminlerin yapılması - Daha önce açıklandı
y_pred = classifier.predict(X_test)

# Karar ağacının grafiği
fig, ax = plt.subplots(figsize=(12, 12))
# Gini değeri 0 ile 1 arasında bir sonuç alır ve sonuç 0'a ne kadar yakında o kadar iyi ayrım yapmış olur.
tree.plot_tree(classifier, fontsize=12,ax=ax, feature_names=["Bki"]);
uzunluk = range(1,15)
error1= []
error2= []
for d in uzunluk:
    classifier= DecisionTreeClassifier(max_depth=d)
    classifier.fit(X_train,y_train)
    error1.append(classifier.score(X_train, y_train))
    error2.append(classifier.score(X_test, y_test))
plt.figure(figsize=(20,5))
plt.plot(uzunluk,error1,label="train")
plt.plot(uzunluk,error2,label="test")
plt.xlabel('Max Depth Değeri')
plt.ylabel('Accuracy Score')
plt.legend()

# Kullanıcıdan veri girdisi alın
bki =int(input("bedenkitle indeksini girin :"))


# Girilen verileri bir veri çerçevesinde toplayın
new_data = pd.DataFrame({'Bki': [bki] })


# Tahmin yapın
y_pred = classifier.predict(new_data)

# Tahmin sonucunu yazdırın
print("Tahmin edilen sınıf:",y_pred)