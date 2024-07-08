Source code:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/content/Housing.csv'
housing_data = pd.read_csv(file_path)

# Data cleaning and exploration
print(housing_data.head())
print(housing_data.describe())
print(housing_data.isnull().sum())

# Convert categorical variables to numeric
categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
housing_data = pd.get_dummies(housing_data, columns=categorical_columns, drop_first=True)
print(housing_data.head())

# Define features and target
X = housing_data.drop('price', axis=1)
y = housing_data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse}')
print(f'R-squared: {r2}')

# Optional: Visualize predictions
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# Example testing
example_input = {
    'area': 3500,
    'bedrooms': 4,
    'bathrooms': 2,
    'stories': 2,
    'mainroad': 1,
    'guestroom': 0,
    'basement': 1,
    'hotwaterheating': 0,
    'airconditioning': 1,
    'parking': 2,
    'prefarea': 1,
    'furnishingstatus': 'semi-furnished'  # Example categorical input
}

# Preprocess example input (convert to DataFrame and encode)
example_input_df = pd.DataFrame([example_input])
example_input_df = pd.get_dummies(example_input_df)
missing_cols = set(X.columns) - set(example_input_df.columns)
for col in missing_cols:
    example_input_df[col] = 0
example_input_df = example_input_df[X.columns]

# Make prediction
example_pred = model.predict(example_input_df)

print("Example Input:", example_input)
print("Predicted House Price: ${:,.2f}".format(example_pred[0]))

!pip install pyTelegramBotAPI
import telebot
# Telegram bot setup
TOKEN = "7223741490:AAE8xSK-0g57Siu4fLdxc27YfHuZork-mYw"
bot = telebot.TeleBot(TOKEN)

# Feature order as in the dataset after get_dummies
feature_order = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
    'mainroad_yes', 'guestroom_yes', 'basement_yes',
    'hotwaterheating_yes', 'airconditioning_yes',
    'prefarea_yes', 'furnishingstatus_semi-furnished',
    'furnishingstatus_unfurnished'
]

@bot.message_handler(commands=['start'])
def send_welcome(message):
    welcome_message = (
        "Welcome to the House Price Prediction Bot!\n"
        "Please enter the required features separated by commas:\n"
        "area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,"
        "hotwaterheating,airconditioning,parking,prefarea,furnishingstatus\n\n"
        "Example: 3500,4,2,2,1,0,1,0,1,2,1,semi-furnished"
    )
    bot.reply_to(message, welcome_message)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_input = message.text
    prediction = predict_house_price(user_input)
    bot.reply_to(message, prediction)

def predict_house_price(user_input: str) -> str:
    features = user_input.split(",")

    if len(features) != 12:
        return f"Error: Please provide 12 features."

    try:
        example_input = {
            'area': float(features[0]),
            'bedrooms': float(features[1]),
            'bathrooms': float(features[2]),
            'stories': float(features[3]),
            'mainroad_yes': int(features[4]),
            'guestroom_yes': int(features[5]),
            'basement_yes': int(features[6]),
            'hotwaterheating_yes': int(features[7]),
            'airconditioning_yes': int(features[8]),
            'parking': float(features[9]),
            'prefarea_yes': int(features[10]),
            'furnishingstatus_semi-furnished': 0,
            'furnishingstatus_unfurnished': 0
        }

        # Handling furnishing status
        if features[11] == 'semi-furnished':
            example_input['furnishingstatus_semi-furnished'] = 1
        elif features[11] == 'unfurnished':
            example_input['furnishingstatus_unfurnished'] = 1

        # Convert to DataFrame and ensure all columns match
        example_input_df = pd.DataFrame([example_input])
        example_input_df = example_input_df.reindex(columns=feature_order, fill_value=0)

        # Make prediction
        example_pred = model.predict(example_input_df)
        return f"The predicted house price is: ₹{example_pred[0]:,.2f}"
    except ValueError as e:
        return f"Error: {str(e)}. Please ensure all inputs are numeric and formatted correctly."

bot.polling()
