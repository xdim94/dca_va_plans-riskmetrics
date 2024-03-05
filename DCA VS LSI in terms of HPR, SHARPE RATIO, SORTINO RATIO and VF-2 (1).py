#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('europedailyprices.csv')

def calculate_sharpe_ratio(final_return, returns):
    # Assuming risk-free rate is zero
    risk_free_rate = 0.0

    # Calculate excess return using the final return
    excess_return = final_return - risk_free_rate

    # Calculate standard deviation of returns
    volatility = returns.std()

    # Calculate Sharpe ratio
    sharpe_ratio = excess_return / (volatility *  12**0.5)

    return sharpe_ratio

def calculate_sortino_ratio(final_return, returns):
    # Assuming risk-free rate is zero
    risk_free_rate = 0.0

    # Calculate downside deviation
    downside_deviation = returns[returns < 0].std()

    # Calculate Sortino ratio using the final return
    sortino_ratio = final_return / (downside_deviation *  12**0.5)

    return sortino_ratio

def kahneman_tversky_valuation(x, alpha=0.88, lambda_=2.25):
    if x >= 0:
        return x ** alpha
    else:
        return -lambda_ * (-x) ** alpha

def dca_strategy(df, index_column, monthly_investment, investment_period):
    # Convert the "Date" column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Create a new column for the number of shares bought
    df['Shares'] = 0.0

    # Initialize a new column for the investment amount
    df['Investment'] = 0.0

    # Initialize a new column for the cumulative invested amount
    df['Invested Amount'] = 0.0

    # Set the monthly investment amount
    monthly_investment = 100

    # Initialize a variable to keep track of accumulated shares
    accumulated_shares = 0.0

    # Iterate through each row and apply DCA strategy for the specified index
    for i, row in df.iterrows():
        # Check if it's the first day of the month and set the investment value
        if i == 0 or row['Date'].month != df['Date'].iloc[i - 1].month:
            df.at[i, 'Investment'] = monthly_investment
            accumulated_shares += df.at[i, 'Investment'] / row[index_column]

        # Assign the accumulated shares to the 'Shares' column
        df.at[i, 'Shares'] = accumulated_shares

        # Update the 'Invested Amount' column
        df.at[i, 'Invested Amount'] = df.at[i, 'Investment'] + df.at[i - 1, 'Invested Amount'] if i > 0 else df.at[i, 'Investment']

    # Calculate the daily portfolio value for the specified index
    df['Portfolio Value'] = df['Shares'] * df[index_column]

    # Calculate daily returns
    df['Daily Returns'] = (df['Portfolio Value'] / df['Invested Amount'] - 1) * 100
    
    # Extract the relevant data for the specified investment period
    start_date = df['Date'].min()
    end_date = start_date + pd.DateOffset(months=investment_period)
    relevant_data = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # Calculate and return the final return
    final_return = relevant_data['Daily Returns'].iloc[-1]
    return final_return, relevant_data['Daily Returns']

def lump_sum_strategy(df, index_column, investment_period):
    # Convert the "Date" column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Set the lump sum investment amount per month
    monthly_investment = 100

    # Calculate the total investment for the specified period
    total_investment = monthly_investment * investment_period

    # Calculate the number of shares bought on the first day
    initial_shares = total_investment / df.loc[0, index_column]

    # Calculate the daily portfolio value for the specified index
    df['Portfolio Value'] = initial_shares * df[index_column]

    # Calculate daily returns
    df['Daily Returns'] = (df['Portfolio Value'] / total_investment - 1) * 100

    # Extract the relevant data for the specified investment period
    start_date = df['Date'].min()
    end_date = start_date + pd.DateOffset(months=investment_period)
    relevant_data = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # Calculate and return the final return
    final_return = relevant_data['Daily Returns'].iloc[-1]
    return final_return, relevant_data['Daily Returns']

# Test different investment periods for DCA (in months)
investment_periods_dca = [6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144]

# Test different investment periods for lump sum (in months)
investment_periods_lump_sum = [6, 12, 24, 36, 48, 60, 72, 84, 96, 108,  120, 132, 144]

# List of index columns
index_columns = ['EUSTX50', 'CAC40', 'DAX', 'IBEX35', 'IT40']

# Store the results in dictionaries
results_dca = {}
results_lump_sum = {}
sharpe_ratios_dca = {}
sharpe_ratios_lump_sum = {}
sortino_ratios_dca = {}
sortino_ratios_lump_sum = {}
valuations_dca = {}
valuations_lump_sum = {}

# Iterate over each index column
for index_column in index_columns:
    # Iterate over each investment period
    for period_dca, period_lump_sum in zip(investment_periods_dca, investment_periods_lump_sum):
        # DCA results
        final_return_dca, daily_returns_dca = dca_strategy(df.copy(), index_column, 100, period_dca)
        results_dca[f'{index_column} - Investment Period {period_dca} Months'] = final_return_dca
        sharpe_ratio_dca = calculate_sharpe_ratio(final_return_dca, daily_returns_dca)
        sharpe_ratios_dca[f'{index_column} - Investment Period {period_dca} Months'] = sharpe_ratio_dca
        sortino_ratio_dca = calculate_sortino_ratio(final_return_dca, daily_returns_dca)
        sortino_ratios_dca[f'{index_column} - Investment Period {period_dca} Months'] = sortino_ratio_dca

        # Lump sum results
        final_return_lump_sum, daily_returns_lump_sum = lump_sum_strategy(df.copy(), index_column, period_lump_sum)
        results_lump_sum[f'{index_column} - Investment Period {period_lump_sum} Months'] = final_return_lump_sum
        sharpe_ratio_lump_sum = calculate_sharpe_ratio(final_return_lump_sum, daily_returns_lump_sum)
        sharpe_ratios_lump_sum[f'{index_column} - Investment Period {period_lump_sum} Months'] = sharpe_ratio_lump_sum
        sortino_ratio_lump_sum = calculate_sortino_ratio(final_return_lump_sum, daily_returns_lump_sum)
        sortino_ratios_lump_sum[f'{index_column} - Investment Period {period_lump_sum} Months'] = sortino_ratio_lump_sum

        # Apply the valuation function to DCA and Lump Sum final returns
        valuation_dca = kahneman_tversky_valuation(final_return_dca)
        valuation_lump_sum = kahneman_tversky_valuation(final_return_lump_sum)

        # Store the valuations in dictionaries
        valuations_dca[f'{index_column} - Investment Period {period_dca} Months'] = valuation_dca
        valuations_lump_sum[f'{index_column} - Investment Period {period_lump_sum} Months'] = valuation_lump_sum

        # Plotting for each investment period
        plt.figure(figsize=(24, 10))

        # Plot DCA strategy returns
        plt.plot(daily_returns_dca.index, daily_returns_dca, label=f'DCA {index_column} - {period_dca} Months', marker='o')

        # Plot lump sum strategy returns
        plt.plot(daily_returns_lump_sum.index, daily_returns_lump_sum, label=f'Lump Sum {index_column} - {period_lump_sum} Months', marker='o')

        # Adding labels and title
        plt.xlabel('Days of Investment')
        plt.ylabel('HPR (%)')
        plt.title(f'{index_column} DCA vs Lump Sum Strategy Returns\nInvestment Period: {period_dca} vs {period_lump_sum} Months')
        plt.legend()
        # Save the plot to a file
        plt.savefig(f'{index_column}_DCA_vs_Lump_Sum_{period_dca}_vs_{period_lump_sum}_Months.png')

        # Show the plot
        plt.show()

# Print or visualize the final returns, Sharpe ratios, and Sortino ratios for each strategy and each time frame
print("DCA Final Returns:")
for period, result in results_dca.items():
    print(f"{period}: {result:.2f}%")

print("\nDCA Sharpe Ratios:")
for period, sharpe_ratio in sharpe_ratios_dca.items():
    print(f"{period}: {sharpe_ratio:.4f}")

print("\nDCA Sortino Ratios:")
for period, sortino_ratio in sortino_ratios_dca.items():
    print(f"{period}: {sortino_ratio:.4f}")

print("\nLump Sum Final Returns:")
for period, result in results_lump_sum.items():
    print(f"{period}: {result:.2f}%")

print("\nLump Sum Sharpe Ratios:")
for period, sharpe_ratio in sharpe_ratios_lump_sum.items():
    print(f"{period}: {sharpe_ratio:.4f}")

print("\nLump Sum Sortino Ratios:")
for period, sortino_ratio in sortino_ratios_lump_sum.items():
    print(f"{period}: {sortino_ratio:.4f}")

print("DCA Valuations:")
for period, valuation in valuations_dca.items():
    print(f"{period}: {valuation:.2f}")

print("\nLump Sum Valuations:")
for period, valuation in valuations_lump_sum.items():
    print(f"{period}: {valuation:.2f}")


# In[24]:


# ... (previous code)

# Iterate over each index column
for index_column in index_columns:
    # Initialize subplots for Sharpe and Sortino ratios
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Initialize lists to store ratios for each index
    sharpe_ratios_dca_index = []
    sharpe_ratios_lump_sum_index = []
    sortino_ratios_dca_index = []
    sortino_ratios_lump_sum_index = []

    # Iterate over each investment period
    for period_dca, period_lump_sum in zip(investment_periods_dca, investment_periods_lump_sum):
        # DCA results
        final_return_dca, daily_returns_dca = dca_strategy(df.copy(), index_column, 100, period_dca)
        sharpe_ratio_dca = calculate_sharpe_ratio(final_return_dca, daily_returns_dca)
        sortino_ratio_dca = calculate_sortino_ratio(final_return_dca, daily_returns_dca)

        # Lump sum results
        final_return_lump_sum, daily_returns_lump_sum = lump_sum_strategy(df.copy(), index_column, period_lump_sum)
        sharpe_ratio_lump_sum = calculate_sharpe_ratio(final_return_lump_sum, daily_returns_lump_sum)
        sortino_ratio_lump_sum = calculate_sortino_ratio(final_return_lump_sum, daily_returns_lump_sum)

        # Append the ratios to the lists for the current index
        sharpe_ratios_dca_index.append(sharpe_ratio_dca)
        sortino_ratios_dca_index.append(sortino_ratio_dca)
        sharpe_ratios_lump_sum_index.append(sharpe_ratio_lump_sum)
        sortino_ratios_lump_sum_index.append(sortino_ratio_lump_sum)

    # Plot Sharpe ratios
    ax1.plot(investment_periods_dca, sharpe_ratios_dca_index, label='DCA', marker='o')
    ax1.plot(investment_periods_lump_sum, sharpe_ratios_lump_sum_index, label='Lump Sum', marker='o')
    ax1.set_title(f'Sharpe Ratios for {index_column}')
    ax1.set_xlabel('Investment Period (Months)')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.legend()

    # Plot Sortino ratios
    ax2.plot(investment_periods_dca, sortino_ratios_dca_index, label='DCA', marker='o')
    ax2.plot(investment_periods_lump_sum, sortino_ratios_lump_sum_index, label='Lump Sum', marker='o')
    ax2.set_title(f'Sortino Ratios for {index_column}')
    ax2.set_xlabel('Investment Period (Months)')
    ax2.set_ylabel('Sortino Ratio')
    ax2.legend()

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()


# In[21]:


import csv

# Define the file path for the CSV file
csv_file_path = 'investment_resultsbreuro.csv'

# Create a list of dictionaries for each result
results_list = []

# Append DCA results
for period, result in results_dca.items():
    result_dict = {'Strategy': 'DCA', 'Period': period, 'Final Return': result}
    result_dict.update({'Sharpe Ratio': sharpe_ratios_dca[period],
                        'Sortino Ratio': sortino_ratios_dca[period],
                        'Valuation': valuations_dca[period]})
    results_list.append(result_dict)

# Append Lump Sum results
for period, result in results_lump_sum.items():
    result_dict = {'Strategy': 'Lump Sum', 'Period': period, 'Final Return': result}
    result_dict.update({'Sharpe Ratio': sharpe_ratios_lump_sum[period],
                        'Sortino Ratio': sortino_ratios_lump_sum[period],
                        'Valuation': valuations_lump_sum[period]})
    results_list.append(result_dict)

# Write results to CSV
with open(csv_file_path, 'w', newline='') as csv_file:
    fieldnames = ['Strategy', 'Period', 'Final Return', 'Sharpe Ratio', 'Sortino Ratio', 'Valuation']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # Write header
    writer.writeheader()
    
    # Write data
    for result_dict in results_list:
        writer.writerow(result_dict)

print(f"Results saved to {csv_file_path}")


# In[22]:




# Combine the results into a DataFrame
results_data = {
    'Strategy': [],
    'Investment Period': [],
    'HPR (%)': [],
    'Sharpe Ratio': [],
    'Sortino Ratio': [],
    'Valuation': []
}

for index_column in index_columns:
    for period_dca, period_lump_sum in zip(investment_periods_dca, investment_periods_lump_sum):
        # DCA results
        final_return_dca, _ = dca_strategy(df.copy(), index_column, 100, period_dca)
        sharpe_ratio_dca = calculate_sharpe_ratio(final_return_dca, daily_returns_dca)
        sortino_ratio_dca = calculate_sortino_ratio(final_return_dca, daily_returns_dca)
        valuation_dca = kahneman_tversky_valuation(final_return_dca)

        # Lump sum results
        final_return_lump_sum, _ = lump_sum_strategy(df.copy(), index_column, period_lump_sum)
        sharpe_ratio_lump_sum = calculate_sharpe_ratio(final_return_lump_sum, daily_returns_lump_sum)
        sortino_ratio_lump_sum = calculate_sortino_ratio(final_return_lump_sum, daily_returns_lump_sum)
        valuation_lump_sum = kahneman_tversky_valuation(final_return_lump_sum)

        # Calculate outperformance percentages
        hpr_outperformance = ((final_return_lump_sum - final_return_dca) / final_return_dca) * 100
        sharpe_outperformance = ((sharpe_ratio_lump_sum - sharpe_ratio_dca) / sharpe_ratio_dca) * 100
        sortino_outperformance = ((sortino_ratio_lump_sum - sortino_ratio_dca) / sortino_ratio_dca) * 100
        valuation_outperformance = ((valuation_lump_sum - valuation_dca) / valuation_dca) * 100

        # Add data to the results_data dictionary
        results_data['Strategy'].extend(['Lump Sum - DCA'] * 4)
        results_data['Investment Period'].extend([f'{period_lump_sum} Months'] * 4)
        results_data['HPR (%)'].extend([hpr_outperformance, 0, 0, 0])
        results_data['Sharpe Ratio'].extend([0, sharpe_outperformance, 0, 0])
        results_data['Sortino Ratio'].extend([0, 0, sortino_outperformance, 0])
        results_data['Valuation'].extend([0, 0, 0, valuation_outperformance])

# Create a DataFrame from the results_data dictionary
results_df = pd.DataFrame(results_data)

# Replace the relevant part of the code with this
# Pivot the DataFrame for better readability
pivot_df = pd.pivot_table(results_df, values=['HPR (%)', 'Sharpe Ratio', 'Sortino Ratio', 'Valuation'],
                          index='Investment Period', columns='Strategy', aggfunc='mean')

# Print the table
print("Outperformance of Lump Sum over DCA:")
print(pivot_df)


# In[19]:


import pandas as pd

# Assuming you have defined functions like dca_strategy, lump_sum_strategy,
# calculate_sharpe_ratio, calculate_sortino_ratio, and kahneman_tversky_valuation

# List of indices to iterate over
indices = ['SZSE GDP100', 'JSE 25', 'NIFTY50', 'IMOEX', 'IBOVESPA']

# Combine the results into a DataFrame
results_data = {
    'Index': [],
    'Strategy': [],
    'Investment Period': [],
    'HPR (%)': [],
    'Sharpe Ratio': [],
    'Sortino Ratio': [],
    'Valuation': []
}

for index_column in indices:
    for period_dca, period_lump_sum in zip(investment_periods_dca, investment_periods_lump_sum):
        # DCA results
        final_return_dca, daily_returns_dca = dca_strategy(df.copy(), index_column, 100, period_dca)
        sharpe_ratio_dca = calculate_sharpe_ratio(final_return_dca, daily_returns_dca)
        sortino_ratio_dca = calculate_sortino_ratio(final_return_dca, daily_returns_dca)
        valuation_dca = kahneman_tversky_valuation(final_return_dca)

        # Lump sum results
        final_return_lump_sum, daily_returns_lump_sum = lump_sum_strategy(df.copy(), index_column, period_lump_sum)
        sharpe_ratio_lump_sum = calculate_sharpe_ratio(final_return_lump_sum, daily_returns_lump_sum)
        sortino_ratio_lump_sum = calculate_sortino_ratio(final_return_lump_sum, daily_returns_lump_sum)
        valuation_lump_sum = kahneman_tversky_valuation(final_return_lump_sum)

        # Calculate outperformance percentages
        hpr_outperformance = ((final_return_lump_sum - final_return_dca) / final_return_dca) * 100
        sharpe_outperformance = ((sharpe_ratio_lump_sum - sharpe_ratio_dca) / sharpe_ratio_dca) * 100
        sortino_outperformance = ((sortino_ratio_lump_sum - sortino_ratio_dca) / sortino_ratio_dca) * 100
        valuation_outperformance = ((valuation_lump_sum - valuation_dca) / valuation_dca) * 100

        # Add data to the results_data dictionary
        results_data['Index'].extend([index_column] * 4)
        results_data['Strategy'].extend(['Lump Sum - DCA'] * 4)
        results_data['Investment Period'].extend([f'{period_lump_sum} Months'] * 4)
        results_data['HPR (%)'].extend([hpr_outperformance, 0, 0, 0])
        results_data['Sharpe Ratio'].extend([0, sharpe_outperformance, 0, 0])
        results_data['Sortino Ratio'].extend([0, 0, sortino_outperformance, 0])
        results_data['Valuation'].extend([0, 0, 0, valuation_outperformance])

# Create a DataFrame from the results_data dictionary
results_df = pd.DataFrame(results_data)

# Pivot the DataFrame for better readability
pivot_df = pd.pivot_table(results_df, values=['HPR (%)', 'Sharpe Ratio', 'Sortino Ratio', 'Valuation'],
                          index=['Index', 'Investment Period'], columns='Strategy', aggfunc='mean')

# Print the table
print("Outperformance of BnH over DCA:")
print(pivot_df)


# In[23]:


# Specify the file path where you want to save the CSV file
output_file_path = 'output_resultsbreuro.csv'

# Save the DataFrame to a CSV file
pivot_df.to_csv(output_file_path)

# Print a message indicating the file has been saved
print(f"Results have been saved to {output_file_path}")


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('IBOVESPA.csv')

def calculate_sharpe_ratio(final_return, returns):
    # Assuming risk-free rate is zero
    risk_free_rate = 0.0

    # Calculate excess return using the final return
    excess_return = final_return - risk_free_rate

    # Calculate standard deviation of returns
    volatility = returns.std()

    # Calculate Sharpe ratio
    sharpe_ratio = excess_return / (volatility *  12**0.5)

    return sharpe_ratio

def calculate_sortino_ratio(final_return, returns):
    # Assuming risk-free rate is zero
    risk_free_rate = 0.0

    # Calculate downside deviation
    downside_deviation = returns[returns < 0].std()

    # Calculate Sortino ratio using the final return
    sortino_ratio = final_return / (downside_deviation *  12**0.5)

    return sortino_ratio

def kahneman_tversky_valuation(x, alpha=0.88, lambda_=2.25):
    if x >= 0:
        return x ** alpha
    else:
        return -lambda_ * (-x) ** alpha

def dca_strategy(df, index_column, monthly_investment, investment_period):
    # Convert the "Date" column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Create a new column for the number of shares bought
    df['Shares'] = 0.0

    # Initialize a new column for the investment amount
    df['Investment'] = 0.0

    # Initialize a new column for the cumulative invested amount
    df['Invested Amount'] = 0.0

    # Set the monthly investment amount
    monthly_investment = 100

    # Initialize a variable to keep track of accumulated shares
    accumulated_shares = 0.0

    # Iterate through each row and apply DCA strategy for the specified index
    for i, row in df.iterrows():
        # Check if it's the first day of the month and set the investment value
        if i == 0 or row['Date'].month != df['Date'].iloc[i - 1].month:
            df.at[i, 'Investment'] = monthly_investment
            accumulated_shares += df.at[i, 'Investment'] / row[index_column]

        # Assign the accumulated shares to the 'Shares' column
        df.at[i, 'Shares'] = accumulated_shares

        # Update the 'Invested Amount' column
        df.at[i, 'Invested Amount'] = df.at[i, 'Investment'] + df.at[i - 1, 'Invested Amount'] if i > 0 else df.at[i, 'Investment']

    # Calculate the daily portfolio value for the specified index
    df['Portfolio Value'] = df['Shares'] * df[index_column]

    # Calculate daily returns
    df['Daily Returns'] = (df['Portfolio Value'] / df['Invested Amount'] - 1) * 100
    
    # Extract the relevant data for the specified investment period
    start_date = df['Date'].min()
    end_date = start_date + pd.DateOffset(months=investment_period)
    relevant_data = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # Calculate and return the final return
    final_return = relevant_data['Daily Returns'].iloc[-1]
    return final_return, relevant_data['Daily Returns']

def lump_sum_strategy(df, index_column, investment_period):
    # Convert the "Date" column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Set the lump sum investment amount per month
    monthly_investment = 100

    # Calculate the total investment for the specified period
    total_investment = monthly_investment * investment_period

    # Calculate the number of shares bought on the first day
    initial_shares = total_investment / df.loc[0, index_column]

    # Calculate the daily portfolio value for the specified index
    df['Portfolio Value'] = initial_shares * df[index_column]

    # Calculate daily returns
    df['Daily Returns'] = (df['Portfolio Value'] / total_investment - 1) * 100

    # Extract the relevant data for the specified investment period
    start_date = df['Date'].min()
    end_date = start_date + pd.DateOffset(months=investment_period)
    relevant_data = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # Calculate and return the final return
    final_return = relevant_data['Daily Returns'].iloc[-1]
    return final_return, relevant_data['Daily Returns']

# Test different investment periods for DCA (in months)
investment_periods_dca = [6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144]

# Test different investment periods for lump sum (in months)
investment_periods_lump_sum = [6, 12, 24, 36, 48, 60, 72, 84, 96, 108,  120, 132, 144]

# List of index columns
index_columns = ['Price']

# Store the results in dictionaries
results_dca = {}
results_lump_sum = {}
sharpe_ratios_dca = {}
sharpe_ratios_lump_sum = {}
sortino_ratios_dca = {}
sortino_ratios_lump_sum = {}
valuations_dca = {}
valuations_lump_sum = {}

# Iterate over each index column
for index_column in index_columns:
    # Iterate over each investment period
    for period_dca, period_lump_sum in zip(investment_periods_dca, investment_periods_lump_sum):
        # DCA results
        final_return_dca, daily_returns_dca = dca_strategy(df.copy(), index_column, 100, period_dca)
        results_dca[f'{index_column} - Investment Period {period_dca} Months'] = final_return_dca
        sharpe_ratio_dca = calculate_sharpe_ratio(final_return_dca, daily_returns_dca)
        sharpe_ratios_dca[f'{index_column} - Investment Period {period_dca} Months'] = sharpe_ratio_dca
        sortino_ratio_dca = calculate_sortino_ratio(final_return_dca, daily_returns_dca)
        sortino_ratios_dca[f'{index_column} - Investment Period {period_dca} Months'] = sortino_ratio_dca

        # Lump sum results
        final_return_lump_sum, daily_returns_lump_sum = lump_sum_strategy(df.copy(), index_column, period_lump_sum)
        results_lump_sum[f'{index_column} - Investment Period {period_lump_sum} Months'] = final_return_lump_sum
        sharpe_ratio_lump_sum = calculate_sharpe_ratio(final_return_lump_sum, daily_returns_lump_sum)
        sharpe_ratios_lump_sum[f'{index_column} - Investment Period {period_lump_sum} Months'] = sharpe_ratio_lump_sum
        sortino_ratio_lump_sum = calculate_sortino_ratio(final_return_lump_sum, daily_returns_lump_sum)
        sortino_ratios_lump_sum[f'{index_column} - Investment Period {period_lump_sum} Months'] = sortino_ratio_lump_sum

        # Apply the valuation function to DCA and Lump Sum final returns
        valuation_dca = kahneman_tversky_valuation(final_return_dca)
        valuation_lump_sum = kahneman_tversky_valuation(final_return_lump_sum)

        # Store the valuations in dictionaries
        valuations_dca[f'{index_column} - Investment Period {period_dca} Months'] = valuation_dca
        valuations_lump_sum[f'{index_column} - Investment Period {period_lump_sum} Months'] = valuation_lump_sum

       # Plotting for each investment period
        plt.figure(figsize=(24, 10))

        # Plot DCA strategy returns
        plt.plot(daily_returns_dca.index, daily_returns_dca, label=f'DCA {index_column} - {period_dca} Months', marker='o')

        # Plot lump sum strategy returns
        plt.plot(daily_returns_lump_sum.index, daily_returns_lump_sum, label=f'Lump Sum {index_column} - {period_lump_sum} Months', marker='o')

        # Adding labels and title
        plt.xlabel('Days of Investment')
        plt.ylabel('HPR (%)')
        plt.title(f' IBOVESPA DCA vs Lump Sum Strategy Returns\nInvestment Period: {period_dca} vs {period_lump_sum} Months')
        plt.legend()

# Print or visualize the final returns, Sharpe ratios, and Sortino ratios for each strategy and each time frame
print("DCA Final Returns:")
for period, result in results_dca.items():
    print(f"{period}: {result:.2f}%")

print("\nDCA Sharpe Ratios:")
for period, sharpe_ratio in sharpe_ratios_dca.items():
    print(f"{period}: {sharpe_ratio:.4f}")

print("\nDCA Sortino Ratios:")
for period, sortino_ratio in sortino_ratios_dca.items():
    print(f"{period}: {sortino_ratio:.4f}")

print("\nLump Sum Final Returns:")
for period, result in results_lump_sum.items():
    print(f"{period}: {result:.2f}%")

print("\nLump Sum Sharpe Ratios:")
for period, sharpe_ratio in sharpe_ratios_lump_sum.items():
    print(f"{period}: {sharpe_ratio:.4f}")

print("\nLump Sum Sortino Ratios:")
for period, sortino_ratio in sortino_ratios_lump_sum.items():
    print(f"{period}: {sortino_ratio:.4f}")

print("DCA Valuations:")
for period, valuation in valuations_dca.items():
    print(f"{period}: {valuation:.2f}")

print("\nLump Sum Valuations:")
for period, valuation in valuations_lump_sum.items():
    print(f"{period}: {valuation:.2f}")


# In[ ]:




