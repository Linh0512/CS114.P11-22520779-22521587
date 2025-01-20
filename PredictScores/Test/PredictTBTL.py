import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# 1. Đọc dữ liệu
# Giả sử dữ liệu được lưu trong file CSV "data.csv"
annonimized = pd.read_csv("Dataset/annonimized.csv.csv")
tbtl = pd.read_excel("Dataset/public_it001/tbtl-public.ods", engine='odf')

# Merge dữ liệu: giữ tất cả các username từ annonimized
data = annonimized.merge(tbtl, on="username", how="left")

# Phân tách dữ liệu
train_test_data = data[data['TBTL'].notna()]  # Sinh viên có điểm, dùng để train
predict_data = data[data['TBTL'].isna()]  # Sinh viên không có điểm, dùng để dự đoán

# 3. Tạo thêm các đặc trưng mới
train_test_data['submission_count'] = train_test_data.groupby('username')['is_final'].transform('count')
train_test_data['avg_pre_score'] = train_test_data.groupby('username')['pre_score'].transform('mean')
train_test_data['pre_score_ratio'] = train_test_data['pre_score'] / (train_test_data['coefficient'] + 1e-5)  # Tránh chia cho 0

# 4. Chọn đặc trưng và biến mục tiêu
features = ['coefficient', 'status_encoded', 'is_final', 'submission_count', 'avg_pre_score', 'pre_score_ratio']

X = train_test_data[features]
y = train_test_data['tbtl']

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 5. Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 6. Huấn luyện mô hình
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

results = {}

for name, model in models.items():
    # Huấn luyện
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Đánh giá
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2
    }

# 7. In kết quả
print("Model Evaluation Results:")
for name, metrics in results.items():
    print(f"{name}: MSE={metrics['MSE']:.4f}, MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")

# 8. Heatmap kiểm tra tương quan
plt.figure(figsize=(10, 8))
sns.heatmap(data[features + [y]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
