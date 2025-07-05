import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

def preprocess_data(df):
    # Drop the 'owner' column
    df = df.drop(columns=['owner'])
    
    # Convert categorical columns to numerical values using one-hot encoding
    df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission'], drop_first=True)
    
    return df

def train_model(df):
    # Define features (X) and target (y)
    X = df.drop(columns=['name'])
    y = df['name']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    
    return model

def save_model(model, filename='car_name_model.pkl'):
    # Save the trained model using pickle
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved as {filename}')

if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Train the model
    model = train_model(df)
    
    # Save the model
    save_model(model)