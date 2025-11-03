import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

# --- CONFIGURATION ---

# FIX: Set TIMESTEPS to 10 to resolve "Not enough training data" error (16 days < 60 timesteps)
DATA_PATH = r"C:\Users\IICET 22\Desktop\abhishek\project store\google stoke price prediction\Google_Stock_Price_Test.csv"
TIMESTEPS = 10 

# --- CACHED FUNCTIONS ---

@st.cache_data
def load_and_preprocess_data(file_path):
    """Loads, cleans, and scales the training data."""
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}. Please check the path and file name.")
        return None, None, None, None 

    # Step 7: Clean and Convert Numeric Columns
    columns_to_clean = ["Open", "High", "Low", "Close", "Volume"]
    for col in columns_to_clean:
        if col in data.columns and data[col].dtype == object:
            data[col] = data[col].str.replace(',', '').astype(float)
    
    # Step 10: Extract 'Open' Column Values
    if 'Open' not in data.columns:
        st.error("The 'Open' column is missing or incorrectly named.")
        return None, None, None, None

    data_set = data.loc[:, ["Open"]].values

    # Step 12: Split the Dataset (using 80% split)
    train_size = int(len(data_set) * 0.8)
    train = data_set[:train_size]
    test = data_set[train_size:]
    
    if len(train) == 0:
        st.error("Training data set is empty after splitting.")
        return None, None, None, None

    train = train.reshape(-1, 1)
    
    # Step 14: Normalize the Training Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train)
    
    return data_set, train_scaled, scaler, test

@st.cache_data
def create_sequences(train_scaled, timesteps):
    """Creates the time-series sequences for training."""
    
    # Check for insufficient data size (prevents IndexError: tuple index out of range)
    if len(train_scaled) < timesteps:
        st.error(f"Not enough training data ({len(train_scaled)} days) for {timesteps} timesteps.")
        return np.empty((0, timesteps, 1)), np.empty((0,)) 

    # Step 16: Create Input Sequences and Targets
    X_train, Y_train = [], []
    
    for i in range(timesteps, len(train_scaled)):
        X_train.append(train_scaled[i - timesteps:i, 0])
        Y_train.append(train_scaled[i, 0])
        
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    
    if X_train.size == 0:
        st.error("X_train sequences are empty. Check your timesteps setting.")
        return np.empty((0, timesteps, 1)), np.empty((0,)) 

    # Reshape X_train for LSTM input: [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, Y_train

@st.cache_resource
def train_lstm_model(X_train, Y_train, timesteps):
    """Builds and trains the LSTM model."""
    
    if X_train.shape[0] == 0:
        st.warning("Cannot train model: Training data is empty.")
        return None

    st.info("Training the LSTM Model (This may take a moment)...")
    
    model = Sequential()
    
    model.add(LSTM(units=50, return_sequences=True, input_shape=(timesteps, 1)))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=1)) 
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X_train, Y_train, epochs=25, batch_size=32, verbose=0) 
    st.success("LSTM Model Training Complete! 🎉")
    return model

# --- MAIN APP FUNCTION ---

def main():
    st.title("📈 Google Stock Price Prediction (LSTM)")
    st.sidebar.header("Configuration")

    timesteps_input = st.sidebar.number_input(
        "Lookback Timesteps (Sequence Length)",
        min_value=10,
        max_value=120,
        # Default value is now 10
        value=TIMESTEPS, 
        step=10
    )

    # 1. Load Data
    data_set, train_scaled, scaler, test = load_and_preprocess_data(DATA_PATH)
    
    if data_set is None:
        return 

    # 2. Create Sequences
    X_train, Y_train = create_sequences(train_scaled, timesteps_input)
    st.sidebar.text(f"X_train Shape: {X_train.shape}")
    
    if X_train.shape[0] == 0:
        return

    # 3. Train Model
    model = train_lstm_model(X_train, Y_train, timesteps_input)
    
    if model is None:
        return

    # --- PREDICTION ---

    # Prepare and Scale Input Data for Prediction
    inputs_start_index = len(data_set) - len(test) - timesteps_input
    
    if inputs_start_index < 0:
         inputs_start_index = 0
         
    inputs = data_set[inputs_start_index:]
    
    inputs = inputs.reshape(-1, 1)
    inputs_scaled = scaler.transform(inputs)

    # Prepare Test Data and Generate Predictions
    X_test = []
    for i in range(timesteps_input, len(inputs_scaled)):
        X_test.append(inputs_scaled[i - timesteps_input:i, 0])
        
    X_test = np.array(X_test)
    
    if X_test.size == 0:
        st.error("Prediction data (X_test) is empty. Cannot generate forecast.")
        return

    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predicted_stock_price_scaled = model.predict(X_test_reshaped, verbose=0)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price_scaled)

    # --- VISUALIZATION ---
    
    st.header("Actual vs. Predicted Google Stock Price (LSTM)")
    st.caption(f"Forecasting {len(test)} steps using a {timesteps_input}-day lookback sequence.")

    fig = plt.figure(figsize=(15, 6))
    plt.plot(test, color='red', label='Actual Google Stock Price')
    plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price (LSTM)') 
    plt.title('Google Stock Price Prediction')
    plt.xlabel('Time (Test Period Days)')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)
    
    # --- PERFORMANCE METRICS ---
    
    rmse = np.sqrt(mean_squared_error(test, predicted_stock_price))
    st.markdown("### Performance Metrics")
    st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:,.4f}")

# Run the app
if __name__ == "__main__":
    main()