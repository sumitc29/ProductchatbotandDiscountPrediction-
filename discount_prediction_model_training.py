# -----------------------------
# 1️⃣ Imports
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 2️⃣ Load dataset
# -----------------------------
df = pd.read_csv("/home/ubuntu/experiments/amazon.csv").fillna("")

#removing highly correlated column 
df.drop('discounted_price', axis=1, inplace = True)

# Convert numerical columns
df['actual_price'] = df['actual_price'].replace('[^0-9.]', '', regex=True).astype(float)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
df['rating_count'] = df['rating_count'].map(lambda x: x.replace(',', ''))
df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce').fillna(0).astype(int)
# Target
y = df['discount_percentage'].replace('[^0-9.]', '', regex=True).astype(float)
## preprocessing for vbelow 
counts = y.value_counts()
y= y.apply(lambda x: x if counts[x] >= 5 else 10.0)


# -----------------------------
# 3️⃣ Feature selection
# -----------------------------
# Select features: categorical, numerical, text
numerical_features = ['actual_price', 'rating', 'rating_count']
categorical_features = ['category']  # maybe split into top-level category
text_features = ['product_name', 'about_product', 'review_content']

# -----------------------------
# 4️⃣ Preprocessing pipelines
# -----------------------------
# Numerical: standard scaling
numerical_transformer = StandardScaler()

# Categorical: one-hot encoding
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Text: TF-IDF
text_transformer = TfidfVectorizer(max_features=5000, stop_words='english')

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('text_name', text_transformer, 'product_name'),
        ('text_about', text_transformer, 'about_product'),
        ('text_review', text_transformer, 'review_content')
    ]
)

# -----------------------------
# 5️⃣ Train-test split
# -----------------------------
X = df[numerical_features + categorical_features + text_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)


#####conclusion : regressioon does not work properly here 
#### Reason: 


# -----------------------------
# 7️⃣ Model 2: XGBoost Regressor
# -----------------------------
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=10,
        random_state=42,
        tree_method='hist'  # efficient for CPU/GPU
    ))
])

xgb_pipeline.fit(X_train, y_train)
y_pred_xgb = xgb_pipeline.predict(X_test)

print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
print("XGBoost R2:", r2_score(y_test, y_pred_xgb))


'''>>> print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
XGBoost RMSE: 14.562101150927374
>>> print("XGBoost R2:", r2_score(y_test, y_pred_xgb))
XGBoost R2: 0.5402863303135186
>>>'''
 
