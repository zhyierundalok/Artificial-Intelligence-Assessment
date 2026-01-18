import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

data_brands = {
    'horsepower': [180, 320, 150, 450, 220, 130, 280, 350, 110, 400,
                   190, 250, 170, 300, 120, 380, 210, 270, 160, 330],
    'engine_type': [1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 
                    1, 2, 1, 2, 1, 2, 2, 2, 1, 2],  # 1=4-cyl, 2=6-cyl, 3=8-cyl
    'fuel_efficiency': [28, 22, 35, 18, 25, 40, 23, 20, 42, 19,
                        27, 24, 32, 21, 38, 19, 26, 23, 34, 20],
    'price_category': [2, 3, 1, 4, 2, 1, 3, 4, 1, 4,
                       2, 3, 2, 3, 1, 4, 2, 3, 2, 3],  # 1=Budget, 2=Mid, 3=Premium, 4=Luxury
    'safety_rating': [4.2, 4.8, 4.0, 4.9, 4.3, 4.1, 4.6, 4.9, 4.0, 4.8,
                      4.3, 4.5, 4.2, 4.7, 4.1, 4.8, 4.4, 4.6, 4.2, 4.7],
    'brand': ['Honda', 'BMW', 'Toyota', 'Mercedes', 'Ford', 'Hyundai', 
              'Audi', 'Porsche', 'Kia', 'Ferrari', 'Nissan', 'Lexus', 
              'Subaru', 'Jaguar', 'Mazda', 'Lamborghini', 'Chevrolet', 
              'Volvo', 'Volkswagen', 'Tesla']
}

df_brands = pd.DataFrame(data_brands)

le_brand = LabelEncoder()
df_brands['brand_encoded'] = le_brand.fit_transform(df_brands['brand'])

X_brand = df_brands[['horsepower', 'engine_type', 'fuel_efficiency', 'price_category', 'safety_rating']]
y_brand = df_brands['brand_encoded']

X_train_brand, X_test_brand, y_train_brand, y_test_brand = train_test_split(
    X_brand, y_brand, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled_brand = scaler.fit_transform(X_train_brand)
X_test_scaled_brand = scaler.transform(X_test_brand)

rf_brand = RandomForestClassifier(n_estimators=100, random_state=42)
rf_brand.fit(X_train_scaled_brand, y_train_brand)
y_pred_rf_brand = rf_brand.predict(X_test_scaled_brand)

print("=" * 60)
print("ALGORITHM 1: PREDICTING CAR BRAND")
print("=" * 60)
print(f"Dataset shape: {df_brands.shape}")
print(f"Unique brands: {df_brands['brand'].nunique()}")
print("\nRandom Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test_brand, y_pred_rf_brand):.2f}")
print("\nClassification Report:")
print(classification_report(y_test_brand, y_pred_rf_brand, 
                           target_names=le_brand.classes_))

def predict_car_brand(horsepower, engine_type, fuel_efficiency, price_category, safety_rating):
    user_input = np.array([[horsepower, engine_type, fuel_efficiency, price_category, safety_rating]])
    user_input_scaled = scaler.transform(user_input)
    prediction_encoded = rf_brand.predict(user_input_scaled)[0]
    predicted_brand = le_brand.inverse_transform([prediction_encoded])[0]
    
    probabilities = rf_brand.predict_proba(user_input_scaled)[0]
    top_3_idx = np.argsort(probabilities)[-3:][::-1]
    
    print(f"\nRecommended Brand: {predicted_brand}")
    print("Top 3 Brand Recommendations:")
    for idx in top_3_idx:
        brand_name = le_brand.inverse_transform([idx])[0]
        probability = probabilities[idx] * 100
        print(f"  - {brand_name}: {probability:.1f}%")
    
    return predicted_brand

print("\n" + "=" * 60)
print("EXAMPLE PREDICTION FOR CUSTOMER REQUIREMENTS:")
print("Requirements: HP=250, 6-cyl engine, Fuel Eff=24, Mid-range price, Safety=4.5")
print("=" * 60)
predict_car_brand(250, 2, 24, 2, 4.5)