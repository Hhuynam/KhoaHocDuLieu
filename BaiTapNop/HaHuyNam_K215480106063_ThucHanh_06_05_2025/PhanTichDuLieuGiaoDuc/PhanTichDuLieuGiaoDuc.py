import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Tải dữ liệu
file_path = r"D:\HocKy3_2024_2025\KhoaHocDuLieu\education_data.csv"  # Đổi đường dẫn nếu cần
df = pd.read_csv(file_path)

# 2. Tạo chỉ số "hiệu suất học tập tổng hợp"
df["performance_score"] = (df["math_score"] + df["science_score"] + df["literature_score"]) / 3 + df["study_hours"] * 0.5

# 3. Kiểm tra ảnh hưởng của hoạt động ngoại khóa bằng ANOVA
low = df[df["extracurricular"] == "low"]["performance_score"]
medium = df[df["extracurricular"] == "medium"]["performance_score"]
high = df[df["extracurricular"] == "high"]["performance_score"]

stat, p_value = stats.f_oneway(low, medium, high)
print("ANOVA p-value:", p_value)
print("Có ảnh hưởng đáng kể!" if p_value < 0.05 else "Không có ảnh hưởng đáng kể.")

# 4. Tạo đặc trưng "Cân bằng học tập"
df["balance_score"] = df[["math_score", "science_score", "literature_score"]].std(axis=1)
df["balanced_learning"] = (df["balance_score"] < df["balance_score"].mean()).astype(int)

# 5. Tạo đặc trưng "Rủi ro học tập"
df["risk_score"] = df["absence_days"] - df["study_hours"]
df["learning_risk"] = (df["risk_score"] > df["risk_score"].mean()).astype(int)

# 6. Trực quan hóa dữ liệu
plt.figure(figsize=(12, 6))
sns.boxplot(x=df["extracurricular"], y=df["performance_score"])
plt.title("Hiệu suất học tập theo mức độ tham gia ngoại khóa")
plt.xlabel("Mức độ tham gia ngoại khóa")
plt.ylabel("Hiệu suất học tập")
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(df["balance_score"], bins=30, kde=True)
plt.title("Phân bố mức độ cân bằng học tập")
plt.xlabel("Độ lệch giữa các môn học")
plt.ylabel("Số lượng sinh viên")
plt.show()

# 7. Xây dựng mô hình SVM để phân loại sinh viên
df["target"] = (df["performance_score"] < df["performance_score"].mean()).astype(int)
X = df[["balanced_learning", "learning_risk"]]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVC(kernel="linear", C=1.0)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
print("Độ chính xác mô hình SVM:", accuracy_score(y_test, y_pred))

# 8. Tự điều chỉnh siêu tham số SVM bằng GridSearchCV
param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "poly", "rbf"]}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Siêu tham số tối ưu:", grid_search.best_params_)
print("Độ chính xác tối ưu:", grid_search.best_score_)
