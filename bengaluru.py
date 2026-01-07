import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("bengaluru_house_prices.csv")
print(data.columns)



# -------------------------------
# 2. Convert total_sqft
# -------------------------------
def convert_sqft_to_num(x):
    if isinstance(x, str):
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        try:
            return float(x)
        except:
            return np.nan
    return x

data['total_sqft'] = data['total_sqft'].apply(convert_sqft_to_num)



# -------------------------------
# 3. Select Correct Columns ✅
# -------------------------------
data = data[['location', 'total_sqft', 'bath','balcony','price']]
data.dropna(inplace=True)

print("Cleaned shape:", data.shape)



# -------------------------------
# 4. Handle Location
# -------------------------------
top_locations = data['location'].value_counts().nlargest(10).index
data['location'] = data['location'].apply(
    lambda x: x if x in top_locations else 'Other'
)

data = pd.get_dummies(data, columns=['location'], drop_first=True)




# -------------------------------
# 5. Split X and y
# -------------------------------
X = data.drop('price', axis=1)
y = data['price']



# -------------------------------
# 6. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# -------------------------------
# 7. Scale & Train
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

cv_score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
print("CV R2:", cv_score.mean())



# -------------------------------
# 8. Evaluate
# -------------------------------
y_pred = model.predict(X_test_scaled)

print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2  :", r2_score(y_test, y_pred))

print("✅ Model trained successfully")





plt.figure(figsize=(6,4))
plt.hist(data['price'], bins=30)
plt.xlabel("Price (Lakhs)")
plt.ylabel("Count")
plt.title("House Price Distribution")
plt.show()


plt.figure(figsize=(6,4))
plt.scatter(data['total_sqft'], data['price'], alpha=0.5)
plt.xlabel("Total Sqft")
plt.ylabel("Price (Lakhs)")
plt.title("Price vs Total Sqft")
plt.show()


plt.figure(figsize=(6,4))
plt.boxplot(
    [data[data['balcony'] == i]['price'] for i in sorted(data['balcony'].unique())],
    labels=sorted(data['balcony'].unique())
)
plt.xlabel("balcony")
plt.ylabel("Price (Lakhs)")
plt.title("Price vs balcony")
plt.show()


plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--'
)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()


plt.figure(figsize=(6,4))
plt.scatter(data['bath'], data['price'], alpha=0.5)
plt.xlabel("Bathrooms")
plt.ylabel("Price (Lakhs)")
plt.title("Bathrooms vs Price")
plt.show()