# 1. IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib

# ==============================
# 2. LOAD DATA
# ==============================

iris = load_iris()

df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['target'] = iris.target

labels = ['Setosa', 'Versicolot', 'Virginica']

print("Dataset preview:\n", df.head())

# ==============================
# 3. EDA (EXPLORATORY DATA ANALYSIS)
# ==============================

#Pairplot
sns.pairplot(df, hue = 'target')
plt.suptitle("Pairplot of iris Dataset", y = 1.02)
plt.show()

#Correlation heatmap
plt.figure()
sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# ==============================
# 4. PREPROCESSING
# ==============================

X = df.drop('target', axis = 1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state= 42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# 5. MODEL TRAINING
# ==============================

models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

# ==============================
# 6. MODEL EVALUATION
# ==============================

print("\n=== MODEL PERFORMANCE ==\n")

for name, model in trained_models.items():
    preds = model.predict(X_test)

    print(f"🔹{name}")
    print("Accuracy: ", accuracy_score(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))

    #Confusion matrix
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot = True, fmt = 'd')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ==============================
# 7. CROSS VALIDATION
# ==============================

print("\n== CROSS VALIDATION ===\n")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv = 5)
    print(f"{name} CV Accuracy: {scores.mean()}")

# ==============================
# 8. HYPERPARAMETER TUNING (KNN)
# ==============================


param_grid = {'n_neighbors' : [3, 5, 7, 9, 11]}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv = 5)
grid.fit(X_train, y_train)

best_knn = grid.best_estimator_

print("\nBest KNN Parameters: ", grid.best_params_)

# ==============================
# 9. FINAL MODEL EVALUATION
# ==============================

final_preds = best_knn.predict(X_test)

print("n=== FINAL MODEL (TUNED KNN) ===")
print("Accuracy:", accuracy_score(y_test, final_preds))
print("Classification Report:\n", classification_report(y_test, final_preds))

# ==============================
# 10. SAVE MODEL
# ==============================

joblib.dump(best_knn, "iris_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n Model saved successfully!")

# ==============================
# 11. PREDICTION FUNCTION
# ==============================

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    model = joblib.load("iris_model.pkl")
    scaler = joblib.load("scaler.pkl")

    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    data = scaler.transform(data)

    prediction = model.predict(data)[0]
    return labels[prediction]

# ==============================
# 12. TEST PREDICTION
# ==============================

print("\n=== SAMPLE PREDICTION ===")
result = predict_species(5.1, 3.5, 1.4, 0.2)
print("Predicted Species:", result)