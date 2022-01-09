
# Importing libraries
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Function for making our neural network
def make_rnn(X):
    layer1 = tf.keras.layers.Input(shape=(X.shape[1], X.shape[2]))
    layer2 = tf.keras.layers.LSTM(units=40)(layer1)
    layer3 = tf.keras.layers.Dense(units=10000)(layer2)
    layer4 = tf.keras.layers.Dense(units=1)(layer3)
    optimizer = tf.keras.optimizers.Adam(lr=0.0002)
    model = tf.keras.Model(inputs=layer1, outputs=layer4)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.summary()
    return model


# Opening the CPI dataset via Pandas
cpi_df = pd.read_csv('data_trimmed/CPI_MONTHLY.csv')
# Opening the BCPI dataset via Pandas
bcpi_df = pd.read_csv('data_trimmed/BCPI_MONTHLY-sd-1972-01-01.csv')
# Opening the monthly BoC assets and liabilities (B1) dataset via Pandas
b1_df = pd.read_csv('data_trimmed/b1_monthly-sd-1935-01-01.csv')

# Getting date indices from the trimmed datasets (used to grab data only from starting time period)
start_date = '1995-01-01'
cpi_date_index = cpi_df[cpi_df['date'] == start_date].index[0]
bcpi_date_index = bcpi_df[bcpi_df['date'] == start_date].index[0]
b1_date_index = b1_df[b1_df['date'] == start_date].index[0]

# Making a new dataframe, for storing the features of interest from the datasets
monetary_data = pd.DataFrame()

# Starting with the B1 data
# Making a date index column (ascending integers)
date_int = list(range(0, len(b1_df['date'][b1_date_index:])))
monetary_data['date_int'] = date_int
# Getting columns of data from B1
monetary_data['tbills'] = b1_df['V36653'][b1_date_index:]      # Treasury bills
monetary_data['GoC_bonds_under3'] = b1_df['V36655'][b1_date_index:]      # GoC bonds under 3 years
monetary_data['GoC_bonds_3to5'] = b1_df['V36656'][b1_date_index:]      # GoC bonds between 3 and 5 years
monetary_data['GoC_bonds_5to10'] = b1_df['V36657'][b1_date_index:]      # GoC bonds between 5 and 10 years
monetary_data['GoC_bonds_over10'] = b1_df['V36658'][b1_date_index:]      # GoC bonds over 10 years
monetary_data['real_return_bonds'] = b1_df['V1160914547'][b1_date_index:]      # Real return bonds
monetary_data['canada_mortgage_bonds'] = b1_df['V1038100657'][b1_date_index:]      # Canada mortgage bonds
monetary_data['advances_members_pc'] = b1_df['V36663'][b1_date_index:]      # Advances to members of Payments Canada
monetary_data['securities_purchased_under_resale'] = b1_df['V36670'][b1_date_index:]      # Securities purchased under
                                                                                       # resale agreements
monetary_data['other_loans_receivables'] = b1_df['V41550172'][b1_date_index:]      # Other loans and receivables
monetary_data['total_assets'] = b1_df['V36651'][b1_date_index:]      # Total assets
monetary_data['notes_in_circulation'] = b1_df['V36672'][b1_date_index:]      # Notes in circulation
monetary_data['GoC_canadian_dollar_deposits'] = b1_df['V36677'][b1_date_index:]      # GoC Canadian dollar deposits
monetary_data['PC_canadian_dollar_deposits'] = b1_df['V41886561'][b1_date_index:]      # Payments Canada Canadian
                                                                                    # dollar deposits
monetary_data['other_canadian_dollar_deposits'] = b1_df['V36681'][b1_date_index:]      # Other Canadian dollar deposits
monetary_data['securities_sold_under_repurchase'] = b1_df['V41886562'][b1_date_index:]      # Securites sold under
                                                                                         # repurchase agreements
monetary_data['capital'] = b1_df['V41886563'][b1_date_index:]                     # Capital
monetary_data['total_capital_liabilities'] = b1_df['V36671'][b1_date_index:]      # Total capital and liabilities

# Moving to the BCPI data
# Getting columns of data from BCPI
# Each value is float64, so it is converted to int64
monetary_data['bcpi'] = np.int64(bcpi_df['M.BCPI'][bcpi_date_index:])      # Total BCPI
monetary_data['ener'] = np.int64(bcpi_df['M.ENER'][bcpi_date_index:])      # BCPI for energy
monetary_data['mtls'] = np.int64(bcpi_df['M.MTLS'][bcpi_date_index:])      # BCPI for metals and minerals
monetary_data['fopr'] = np.int64(bcpi_df['M.FOPR'][bcpi_date_index:])      # BCPI for forestry
monetary_data['agri'] = np.int64(bcpi_df['M.AGRI'][bcpi_date_index:])      # BCPI for agriculture
monetary_data['fish'] = np.int64(bcpi_df['M.FISH'][bcpi_date_index:])     # BCPI for fish

# Moving to the CPI data
# Getting the total CPI (what the neural network will predict)
monetary_data['total_cpi'] = np.int64(cpi_df['V41690973'][cpi_date_index:])     # Total CPI

# If there are any empty (NAN) values, fill them in with zeros
monetary_data = monetary_data.fillna(0)
# Saving the monetary data as a CSV
monetary_data.to_csv('monetary_data.csv')

# Splitting the monetary data into training and test data
# Since the data is used sequentially, the order of the data remains unchanged
# First make a min-max scaler, and scale the columns of monetary data
data_scaler = MinMaxScaler(feature_range=(-1, 1))
monetary_data_scaled = data_scaler.fit_transform(monetary_data)
# proportion_training is the proportion of data from monetary_data that will
# be put into training data. The rest goes into test data
proportion_training = 0.95
number_months_train = int(np.ceil(len(monetary_data) * proportion_training))
number_months_test = int(len(monetary_data) - number_months_train)
print("Total number of months: " + str(len(monetary_data)))
print("Number of training months: " + str(number_months_train))
print("Number of testing months: " + str(number_months_test))
monetary_data_scaled_train = monetary_data_scaled[:number_months_train]
monetary_data_scaled_test = monetary_data_scaled[number_months_train:]
print("Dimensions of monetary_data_scaled_train: " + str(monetary_data_scaled_train.shape))
print("Dimensions of monetary_data_scaled_test: " + str(monetary_data_scaled_test.shape))
print("Number of elements in monetary_data_scaled_train: " + str(monetary_data_scaled_train.size))
print("Number of elements in monetary_data_scaled_test: " + str(monetary_data_scaled_test.size))
number_features = monetary_data_scaled_train.shape[1] - 1
print("Number of features: " + str(number_features))

# Splitting the training and test data into associated features and labels
# number_prediction_months gives how many months (in sequence) are provided to the
# neural network at any instance, with the network predicting the value for the next month
number_prediction_months = 6
# X_train and X_test are 3-dimensional arrays. Each row represents a training example for the
# neural network, each column represents each month, while each depth represents each feature from monetary_data
# Generating X_train and y_train
X_train = np.zeros((number_months_train-number_prediction_months, number_prediction_months, number_features))
y_train = np.zeros((number_months_train-number_prediction_months))
for i in range(0, number_features):
    for j in range(0, number_months_train-number_prediction_months):
        for k in range(0, number_prediction_months):
            X_train[j][k][i] = monetary_data_scaled_train[j+k, i]
        y_train[j] = monetary_data_scaled_train[j+number_prediction_months, number_features]
print("Dimensions of X_train: " + str(X_train.shape))
print("Dimensions of y_train: " + str(y_train.shape))
# Generating X_test and y_test
X_test = np.zeros((number_months_test-number_prediction_months, number_prediction_months, number_features))
y_test = np.zeros((number_months_test-number_prediction_months))
for i in range(0, number_features):
    for j in range(0, number_months_test-number_prediction_months):
        for k in range(0, number_prediction_months):
            X_test[j][k][i] = monetary_data_scaled_test[j+k, i]
        y_test[j] = monetary_data_scaled_test[j+number_prediction_months, number_features]
print("Dimensions of X_test: " + str(X_test.shape))
print("Dimensions of y_test: " + str(y_test.shape))

# Generate the neural network for analyzing the data
model_rnn = make_rnn(X_train)
# Training our model, using the training dataset
model_rnn.fit(X_train, y_train, epochs=10000)
# Predict the next CPI value, using the test dataset
y_pred = model_rnn.predict(X_test)
y_pred = y_pred[:, 0]
print("Dimensions of y_pred: " + str(y_pred.shape))

# Making vectors for x values, used in plotting
x_train = list(range(0, number_months_train))
x_test = list(range(number_months_train, number_months_train + number_months_test))
x_pred = list(range(number_months_train + number_months_test - len(y_pred) + 1,
                    number_months_train + number_months_test + 1))

# Performing inverse transform on the scaled monetary data
monetary_data_train = data_scaler.inverse_transform(monetary_data_scaled_train)
monetary_data_test = data_scaler.inverse_transform(monetary_data_scaled_test)
# Since the scaler requires number_features+1 columns in order to scale, we will make a
# matrix that has this number of columns, and place y_train (or y_test or y_pred) into
# the last column of the matrix
y_pred_emptymatrix = np.zeros(shape=(len(y_pred), number_features+1))
y_pred_emptymatrix[:, -1] = y_pred
# Now inverse scaling on these matrices, and taking the last column of each
y_pred_emptymatrix = data_scaler.inverse_transform(y_pred_emptymatrix)
y_pred = y_pred_emptymatrix[:, -1]

# Plotting the training, test, and predicted CPI
fig, ax = plt.subplots()
ax.plot(x_train, monetary_data_train[:, -1], color='red')
ax.plot(x_test, monetary_data_test[:, -1], color='black')
ax.plot(x_pred, y_pred, color='blue')
plt.show()
