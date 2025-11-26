# Decision tree model implementation
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")
y_test = pd.read_csv("data/y_test.csv")

model = DecisionTreeRegressor(max_depth=7, random_state=42) # using regression tree because rating is continuous not categories
model.fit(X_train, y_train.values.ravel())
# i started with 10 depth but it was overfitting so i found 7 was best for lowest rmse

y_pred = model.predict(X_test)

# using rmse to evaluate (we can change this i just saw in the book that its what they use for regression idk)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# we want it to be as low as possible because its how many points off it is on avg

print("RMSE:", rmse)

importances = pd.Series(model.feature_importances_, index=X_train.columns)
threshold = 0.003   # dont show the throw away features
important_feats = importances[importances > threshold].sort_values(ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(important_feats.index, important_feats.values)
plt.xlabel("Feature Importance")
plt.title("Important Features")
plt.tight_layout()
plt.show()
