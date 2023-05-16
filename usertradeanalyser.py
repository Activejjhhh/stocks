import pickle


name = input("hello")
print(name)


# Load the saved machine learning model
with open('C://Users//Danie//OneDrive//Desktop//Sublime Coding//model.pkl', 'rb') as f:
    model = pickle.load(f)
# Define a function to get user input for trade characteristics
def get_trade_input():
    trade_input = input('Please enter the following characteristics of your trade: Price, Signal (buy or sell), Stop Loss, Take Profit, Open, High, Low, Close, Volume, Position Size (separated by commas): ')
    trade_input = trade_input.split(',')
    price = float(trade_input[0].strip())
    signal = trade_input[1].strip()
    stop_loss = float(trade_input[2].strip())
    take_profit = float(trade_input[3].strip())
    open_price = float(trade_input[4].strip())
    high = float(trade_input[5].strip())
    low = float(trade_input[6].strip())
    close = float(trade_input[7].strip())
    volume = float(trade_input[8].strip())
    position_size = float(trade_input[9].strip())
    return [price, signal, stop_loss, take_profit, open_price, high, low, close, volume, position_size]


# Define a function to analyze the trade using the machine learning model
def analyze_trade(trade_input):
    # Convert the signal to a binary variable (0 or 1)
    signal = 1 if trade_input[1] == 'buy' else 0
    # Reshape the trade input for use with the machine learning model
    trade_input = [trade_input[0], signal, trade_input[2], trade_input[3], trade_input[4], trade_input[5], trade_input[6], trade_input[7], trade_input[8]]
    trade_input = [trade_input]
    # Use the machine learning model to make a prediction
    prediction = model.predict(trade_input)
    # Interpret the prediction and provide feedback
    if prediction == 1:
        print('Based on the characteristics of your trade, our analysis indicates that this is a good breakout trade.')
        print('Here are some reasons why: ')
        # Insert specific reasons here based on the machine learning model's analysis
    else:
        print('Based on the characteristics of your trade, our analysis indicates that this is not a good breakout trade.')
        print('Here are some reasons why: ')
        # Insert specific reasons here based on the machine learning model's analysis

# Main program loop
print('Welcome to the Breakout Trade Analysis Program!\n')
while True:
    trade_input = get_trade_input()
    print('\nPlease wait while we analyze your trade...\n')
    analyze_trade(trade_input)
    again = input('Would you like to analyze another trade? (y/n): ')
    if again.lower() != 'y':
        break