import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Tải dataset Diabetes từ scikit-learn
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df["target"] = diabetes.target  # Thêm cột mục tiêu

# 2. Khám phá dữ liệu
print(df.head())
print(df.info())
print(df.describe())

# 3. Kiểm tra sự tương quan giữa các đặc trưng
corr_matrix = df.corr()
plt.figure(figsize=(12, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Ma trận tương quan giữa các đặc trưng")
plt.show()

# 4. Chia dữ liệu thành tập huấn luyện và kiểm tra
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Huấn luyện mô hình Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("Linear Regression - R²:", r2_score(y_test, y_pred_lr))
print("Linear Regression - RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))

# 7. Huấn luyện mô hình Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest - R²:", r2_score(y_test, y_pred_rf))
print("Random Forest - RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))

# 8. Tối ưu siêu tham số Random Forest bằng GridSearchCV
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10]
}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Siêu tham số tối ưu:", grid_search.best_params_)
print("Độ chính xác tối ưu - R²:", grid_search.best_score_)

# 9. Trực quan hóa kết quả dự đoán
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_rf)
plt.xlabel("Giá trị thực tế")
plt.ylabel("Giá trị dự đoán")
plt.title("Biểu đồ phân tán - Random Forest")
plt.show()
