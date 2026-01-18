data_types = {
    'horsepower': [120, 180, 220, 300, 150, 250, 400, 90, 280, 350,
                   130, 200, 320, 110, 380, 170, 450, 95, 230, 310],
    'engine_type': [1, 1, 2, 2, 1, 2, 3, 1, 2, 3,
                    1, 2, 2, 1, 3, 1, 3, 1, 2, 2],  # 1=4-cyl, 2=6-cyl, 3=8-cyl
    'fuel_efficiency': [35, 28, 25, 22, 32, 23, 18, 40, 24, 20,
                        34, 26, 21, 38, 19, 30, 17, 42, 27, 22],
    'price': [20000, 28000, 35000, 50000, 25000, 42000, 80000, 18000, 
              45000, 65000, 22000, 32000, 55000, 19000, 75000, 27000, 
              90000, 17000, 38000, 58000],
    'seating_capacity': [5, 5, 5, 5, 7, 5, 2, 5, 5, 4,
                         5, 5, 5, 5, 2, 7, 2, 5, 5, 4],
    'cargo_space': [15, 18, 16, 12, 25, 14, 8, 14, 15, 10,
                    16, 17, 13, 13, 9, 28, 7, 12, 16, 11],
    'car_type': ['Sedan', 'Sedan', 'SUV', 'SUV', 'Minivan', 'Coupe', 'Sports',
                 'Hatchback', 'Sedan', 'Luxury', 'Hatchback', 'SUV', 'Luxury',
                 'Hatchback', 'Sports', 'Minivan', 'Sports', 'Hatchback',
                 'Sedan', 'Luxury']
}

df_types = pd.DataFrame(data_types)

le_type = LabelEncoder()
df_types['type_encoded'] = le_type.fit_transform(df_types['car_type'])

X_type = df_types[['horsepower', 'engine_type', 'fuel_efficiency', 'price', 
                   'seating_capacity', 'cargo_space']]
y_type = df_types['type_encoded']

X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(
    X_type, y_type, test_size=0.2, random_state=42
)

scaler_type = StandardScaler()
X_train_scaled_type = scaler_type.fit_transform(X_train_type)
X_test_scaled_type = scaler_type.transform(X_test_type)

knn_type = KNeighborsClassifier(n_neighbors=3)
knn_type.fit(X_train_scaled_type, y_train_type)
y_pred_knn_type = knn_type.predict(X_test_scaled_type)

print("\n" + "=" * 60)
print("ALGORITHM 2: PREDICTING CAR TYPE/CLASSIFICATION")
print("=" * 60)
print(f"Dataset shape: {df_types.shape}")
print(f"Unique car types: {df_types['car_type'].nunique()}")
print(f"Available types: {list(le_type.classes_)}")
print("\nK-Nearest Neighbors Performance:")
print(f"Accuracy: {accuracy_score(y_test_type, y_pred_knn_type):.2f}")
print("\nClassification Report:")
print(classification_report(y_test_type, y_pred_knn_type, 
                           target_names=le_type.classes_))

def predict_car_type(horsepower, engine_type, fuel_efficiency, price, 
                     seating_capacity, cargo_space):
    user_input = np.array([[horsepower, engine_type, fuel_efficiency, price, 
                           seating_capacity, cargo_space]])
    user_input_scaled = scaler_type.transform(user_input)
    prediction_encoded = knn_type.predict(user_input_scaled)[0]
    predicted_type = le_type.inverse_transform([prediction_encoded])[0]
    
    distances, indices = knn_type.kneighbors(user_input_scaled)
    similar_cars = df_types.iloc[indices[0]]
    
    print(f"\nRecommended Car Type: {predicted_type}")
    print("\nSimilar vehicles in database:")
    for idx, car in similar_cars.iterrows():
        print(f"  - {car['car_type']} (HP: {car['horsepower']}, "
              f"Price: ${car['price']:,.0f}, Seats: {car['seating_capacity']})")
    
    return predicted_type

print("\n" + "=" * 60)
print("EXAMPLE PREDICTION FOR CUSTOMER REQUIREMENTS:")
print("Requirements: HP=220, 6-cyl, Fuel Eff=25, Price=$35,000, Seats=5, Cargo=16")
print("=" * 60)
predicted_type = predict_car_type(220, 2, 25, 35000, 5, 16)

print("\n" + "=" * 60)
print("COMBINED RECOMMENDATION SYSTEM")
print("=" * 60)
print("Based on the two models, we can create a hybrid recommendation:")

def get_car_recommendation():
    print("\nEnter your car requirements:")
    hp = float(input("Horsepower: "))
    engine = int(input("Engine Type (1=4-cyl, 2=6-cyl, 3=8-cyl): "))
    fuel_eff = float(input("Fuel Efficiency (MPG): "))
    price_cat = int(input("Price Category (1=Budget, 2=Mid, 3=Premium, 4=Luxury): "))
    safety = float(input("Safety Rating (1-5): "))
    price = float(input("Budget Price ($): "))
    seats = int(input("Seating Capacity: "))
    cargo = float(input("Cargo Space (cu ft): "))
    
    print("\n" + "=" * 60)
    print("YOUR CAR RECOMMENDATION")
    print("=" * 60)
    
    brand = predict_car_brand(hp, engine, fuel_eff, price_cat, safety)
    
    car_type = predict_car_type(hp, engine, fuel_eff, price, seats, cargo)
    
    print(f"\nFINAL RECOMMENDATION:")
    print(f"Consider a {brand} {car_type}")
    print(f"\nThis recommendation is based on:")
    print(f"- Horsepower: {hp}")
    print(f"- Engine: {'4-cylinder' if engine == 1 else '6-cylinder' if engine == 2 else '8-cylinder'}")
    print(f"- Fuel Efficiency: {fuel_eff} MPG")
    print(f"- Budget: ${price:,.0f}")
    print(f"- Seating: {seats} people")
    
    return brand, car_type