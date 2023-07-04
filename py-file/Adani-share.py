# %% [markdown]
# # Adani transmission stock

# %% [markdown]
# **Example: -1** Portfolio Risk for my investment in Adani Transmission share price 

# %%
# installing yfinance
%pip install yfinance -q

# %%
# importing important libraries
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Get historical price data for Adani Transmission

ticker = 'ADANITRANS.NS'  # Ticker symbol for Adani Transmission
start_date = '2022-01-01'  # Start date for the historical data
end_date = '2023-04-30'  # End date for the historical data

adani_df0 = yf.download(ticker, start=start_date, end=end_date)
adani_df0.head()

# %%
adani_df0.dtypes

# %%
adani_df0.columns

# %%
# checking missing values i.e. NaN values
adani_df0.isnull().sum()

# %%
# shape of the dataframe
adani_df0.shape

# %%
#basic statistics
adani_df0.describe()

# %%
adani_df0.max()

# %%
adani_df0.min()

# %%
# If date is not datetime format then Convert the 'Date' column to datetime type using
# adani_df['Date'] = pd.to_datetime(adani_df['Date'])

# Set the figure size
plt.figure(figsize=(10, 6))
# Plot 'Open' price
sns.lineplot(x='Date', y='Open', data=adani_df0, label='Open')

# Plot 'Close' price
sns.lineplot(x='Date', y='Close', data=adani_df0, label='Close')

# Plot 'High' price
sns.lineplot(x='Date', y='High', data=adani_df0, label='High')

# Plot 'Low' price
sns.lineplot(x='Date', y='Low', data=adani_df0, label='Low')

# Set the labels and title
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Adani Share Data')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Display the legend
plt.legend()

# Show the plot
plt.show()

# %%
# we can also plot all four in separate subfigures in a single figure

# For duration 1996-2023
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
adani_df0['Open'].plot(ax=ax[0,0], title= 'Open') # type: ignore
adani_df0['High'].plot(ax=ax[0,1], title= 'High') # type: ignore
adani_df0['Low'].plot(ax=ax[1,0], title= 'Low') # type: ignore
adani_df0['Close'].plot(ax=ax[1,1], title= 'Close') # type: ignore
# Adjust spacing between subplots
fig.subplots_adjust(wspace=0.5, hspace=0.5)

# %%
# we can also plot all four in separate subfigures in a single figure

# For duration 1996-2023
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
adani_df0['Open'].loc['2023-01-01':'2023-06-01'].plot(ax=ax[0,0], title= 'Open') # type: ignore
adani_df0['High'].loc['2023-01-01':'2023-06-01'].plot(ax=ax[0,1], title= 'High') # type: ignore
adani_df0['Low'].loc['2023-01-01':'2023-06-01'].plot(ax=ax[1,0], title= 'Low') # type: ignore
adani_df0['Close'].loc['2023-01-01':'2023-06-01'].plot(ax=ax[1,1], title= 'Close') # type: ignore
# Adjust spacing between subplots
fig.subplots_adjust(wspace=0.5, hspace=0.5)

# %% [markdown]
# so in last 6 months, Adani Transmission share did not perform well for this period of time.

# %%
# Plot histogram
sns.histplot(data = adani_df0['Close'], kde = True) 
plt.show()

# %%
# Create subplots with 2 rows and 1 column
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot 'Close' vs index
axs[0].plot(adani_df0.index, adani_df['Close'], color='steelblue')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Close')
axs[0].set_title('Close vs Index')
axs[0].grid(True)

# Plot histogram for 'Close'
axs[1].hist(adani_df0['Close'], bins=30, color='lightgreen', edgecolor='black')
axs[1].set_xlabel('Close')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Histogram of Close')
axs[1].grid(True)

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.4)

# Show the plot
plt.show()

# %% [markdown]
# ## 1. Daily changes & % changes
# 
# Here in this section, we will do our calculation in a copied dataframe, so that our original dataframe remain intact while doing these calculations for our future references.
# 
# - Daily change $\Rightarrow $ Daily Change = Current Value - Previous Value
#     
#     - 'Current Value' represents the value of a data point at a specific time (e.g., the closing price of a stock on a particular day).
#     - 'Previous Value' represents the value of the same data point at the previous time (e.g., the closing price of the stock on the previous day).
# 

# %%
# copying original dataframe

adani_df = adani_df0.copy()
adani_df.head()

# %%
# daily change = Open price - Closing price
# Calculate daily change
adani_df['Daily Change'] = adani_df['Close'].diff()

# Calculate percentage daily change
adani_df['% Daily Change'] = adani_df['Close'].pct_change() * 100

adani_df.head()

# %%
# If date is not datetime format then Convert the 'Date' column to datetime type using
# adani_df['Date'] = pd.to_datetime(adani_df['Date'])

# Set the figure size
plt.figure(figsize=(10, 6))

sns.lineplot(x='Date', y='Daily Change', data=adani_df, label='Daily Change')
sns.lineplot(x='Date', y='% Daily Change', data=adani_df, label='% Daily Change')

# Set the labels and title
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Adani Share Data')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Display the legend
plt.legend()

# Show the plot
plt.show()

# %% [markdown]
# ## 2. Retrun
# 
# Returns, in the context of finance and investing, refer to the financial gain or loss on an investment or asset over a specific period. - It measures the percentage change in the value of the investment relative to the initial investment or the previous value.
# - Returns are commonly used to evaluate the performance of investments, compare investment options, and assess the profitability of trading strategies. 
#   - returns > 0 $\Rrightarrow$ profit or gain
#   - returns < 0 $\Rrightarrow$ loss.
# - There are various types of returns depending on the specific calculation and time period considered. Some common types of returns include:
# 
#   1. **Total Return:** It measures the overall return on an investment, taking into account both price appreciation (capital gains) and any income generated (such as dividends or interest).
#   
#       $\boxed{{\rm Total ~Return} = \frac{{\rm Ending ~ Value} - {\rm Beginning ~ Value} + {\rm Income}}{{\rm Beginning ~ Value}}}$
# 
#       where:
# 
#       - `Ending Value` is the value of the investment at the end of the period.
#       - `Beginning Value` is the value of the investment at the beginning of the period.
#       - `Income` refers to any additional income generated during the period, such as dividends or interest.
# 
#   2. **Price Return:** Also known as capital gain or price appreciation, it represents the percentage change in the price or value of an investment over a specified time period.
# 
#       $\boxed{\text{Price Return} = \frac{\text{Ending Price} - \text{Beginning Price}}{\text{Beginning Price}}}$
# 
#       where:
# 
#       - `Ending Price` is the price of the investment at the end of the period.
#       - `Beginning Price` is the price of the investment at the beginning of the period.
# 
#   3. **Dividend Yield:** It measures the return generated by dividends relative to the investment cost. Dividend yield is usually expressed as a percentage.
#   
#       $\boxed{\text{Dividend Yield} = \frac{\text{Dividends}}{\text{Price}} \times 100}$
# 
#       where:
# 
#       - `Dividends` represent the total dividends received during the period.
#       - `Price` is the price of the investment at the beginning or end of the period.
# 
#     To calculate dividends, you need information about the dividend payments made by the company. Typically, dividend payments are announced by the company and are paid out to shareholders at specific intervals.
# 
#   4. **Total Shareholder Return (TSR):** It combines both price appreciation and dividends to calculate the total return to a shareholder over a specific period, including both capital gains and income.
# 
#       $\boxed{\text{TSR} = \frac{\text{Ending Price} + \text{Dividends} - \text{Beginning Price}}{\text{Beginning Price}}}$
# 
#       where:
# 
#       - `Ending Price` is the price of the investment at the end of the period.
#       - `Beginning Price` is the price of the investment at the beginning of the period.
#       - `Dividends` represent the total dividends received during the period.

# %% [markdown]
# ## 3. Price return
# 
# Price return measures the change in the price of an investment over a specific period, usually expressed as a percentage. It is calculated by taking the difference between the ending price and the initial price and dividing it by the initial price. Price return reflects the capital appreciation or depreciation of the investment but does not consider any additional income generated by the investment, such as dividends or interest.
# 
# **Formula:** $\text{Price return} = \left(\frac{\text{Ending Price - Initial Price}}{\text{Initial Price}}\right) \times 100= $
# 
# #### Difference with Total Return
# Total return, on the other hand, considers both the price appreciation and any additional income generated by the investment, such as dividends, interest, or other distributions. It provides a more comprehensive measure of the overall return on investment. Total return takes into account not only the capital gains or losses from changes in price but also the income component generated during the investment period.
# 
# **Formula:** $\text{Total return} = \left(\frac{\text{Ending Value - Initial Value + Income}}{\text{Initial Value}}\right)\times 100$
# 
# 
# - when no income or no dividened is considered: `TSR = Total return = Price return`
# 
# - Retrun $R_i$ % = $\frac{C_i -C_0}{C_0} \times 100$  (when no income is considered).
# 
#     where 
# 
#     - $C_i$ = ith entry
#     - $C_0$ = first entry of the table in that particular coloumn
# 
# #### We can calculate these from following table (no income):
# | Date | Close |    Open  |   High |   Low |
# |------|-------|----------|--------|-------|
# | D1   | $C_0$ |   $O_0$  |  $H_0$ | $L_0$ |
# | D1   | $C_1$ |   $O_1$  |  $H_1$ | $L_1$ |
# | D1   | $C_2$ |   $O_2$  |  $H_2$ | $L_2$ |
# | D1   | $C_3$ |   $O_3$  |  $H_3$ | $L_3$ |
# |  .   |   .   |     .    |    .   |   .   | 
# |  .   |   .   |     .    |    .   |   .   |
# |  Dn  | $C_n$ |   $O_n$  |  $H_n$ | $L_n$ |
# 
# We can calculate total return and price return from this table as follows:
# 
# $TR = \left(\frac{C_n - C_0}{C_0}\right) \times 100$
# 
# Price Return:
# $PR = \left(\frac{C_n - C_0}{C_0}\right) \times 100$
# 
# 
# | Date | Close |    Open  |   High |   Low |        Total Return($\%$)        |         Price Return($\%$)        |
# |------|-------|----------|--------|-------|----------------------------------|-----------------------------------|
# | D1   | $C_0$ |   $O_0$  |  $H_0$ | $L_0$ |              0.0                 |                  0.0              |
# | D1   | $C_1$ |   $O_1$  |  $H_1$ | $L_1$ | $((C_1 - O_1) / O_1) \times 100$ | $((C_1 - C_0) / C_0) \times 100$  |
# | D1   | $C_2$ |   $O_2$  |  $H_2$ | $L_2$ | $((C_2 - O_2) / O_2) \times 100$ | $((C_2 - C_0) / C_0) \times 100$  |
# | D1   | $C_3$ |   $O_3$  |  $H_3$ | $L_3$ | $((C_3 - O_3) / O_3) \times 100$ | $((C_3 - C_0) / C_0) \times 100$  |
# |  .   |   .   |     .    |    .   |   .   |                   .              |                .                  |
# |  .   |   .   |     .    |    .   |   .   |                   .              |                .                  |
# |  Dn  | $C_n$ |   $O_n$  |  $H_n$ | $L_n$ | $((C_n - O_n) / O_n) \times 100$ | $((C_n - C_0) / C_0) \times 100$  |

# %%
# Set the figure size
plt.figure(figsize=(8, 4))

# Set the index of the DataFrame as the 'Date' column
#adani_df.set_index('Date', inplace=True)

# Plot price return using formula
price_return_formula = ((adani_df['Close'] - adani_df['Close'].iloc[0]) / adani_df['Close'].iloc[0]) * 100
plt.plot(adani_df.index, price_return_formula, label='Price Return (Formula)', color='blue')

# Plot price return using pct_change()
price_tot_return = ((adani_df['Close'] - adani_df['Open'] )/ adani_df['Open'])*100
plt.plot(adani_df.index, price_tot_return, label='Price Return (pct_change)', color='green')

# Set the labels and title
plt.xlabel('Date')
plt.ylabel('Price Return (%)')
plt.title('Adani Share Price Return')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Display the legend
plt.legend()

# Show the plot
plt.show()

# %%
# Set the figure size
plt.figure(figsize=(10,4))

# Set the index of the DataFrame as the 'Date' column
#adani_df.set_index('Date', inplace=True)

# Plot price return using formula
price_return_formula = ((adani_df['Close'] - adani_df['Close'].iloc[0]) / adani_df['Close'].iloc[0]) * 100
plt.plot(adani_df.index, price_return_formula, label='Price Return (Formula)', color='blue')

# Plot price return using pct_change()
price_return_pct_change = adani_df['Close'].pct_change() * 100
plt.plot(adani_df.index, price_return_pct_change, label='Price Return (pct_change)', color='green')

# Set the labels and title
plt.xlabel('Date')
plt.ylabel('Price Return (%)')
plt.title('Adani Share Price Return')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Display the legend
plt.legend()

# Show the plot
plt.show()

# %% [markdown]
# so in absence of information of dividened, all the three types of return aare same. So we have
# 
# `adnai_df['Total Return'] = ((C_i - O_1) / O_1) * 100  =adani_df.pct_change()`

# %%
adani_df.head()

# %%
# Adding Total Return and Price return to the adani_df

# Create a new dataframe with selected columns
adani_df1 = adani_df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()

# Calculate and add the Total Return column
adani_df1['Total Return'] = ((adani_df['Close'] - adani_df['Open']) / adani_df['Open']) * 100

# Calculate and add the Price Return column
adani_df1['Price Return'] = ((adani_df['Close'] - adani_df['Close'].iloc[0]) / adani_df['Close'].iloc[0]) * 100

adani_df1.head()

# %%
adani_df1.tail()

# %% [markdown]
# ## 4. Voltality of the market
# 
# - Market volatility refers to the degree of price variability or fluctuations in the overall financial market. 
# - It represents the rate at which market prices change over a given period, indicating the level of uncertainty and risk in the market.
# - The most widely used measure of market volatility is the calculation of the standard deviation of the index returns.
# - The formula for calculating volatility is as follows:
# 
#     $\boxed{\text{Volatility} = \sqrt{\frac{\sum{(R_i - R_{\rm avg})^2}}{(n-1)}} = \sqrt{\text{Variance}}}$
# 
#     Where:
# 
#     - $R_i$ is the return for a given day or time period
#     - $R_{\rm avg}$ is the average return over the same period
#     - $n$ is the number of days or time periods being analyzed
# 
# - Variance is the average of the squared differences between each day's return and the average return over the same period. 
# - Volatility is a measure of the degree of variation or dispersion in the price of a financial instrument, typically represented as a percentage. It quantifies the magnitude of price fluctuations and reflects the level of risk associated with an investment.
# - The formula can also be simplified as follows:
# 
#     $\boxed{\text{Volatility} = \sqrt{\text{Variance}} = \sqrt{252} \times \text{Standard deviation}}$
# 
#     Where:
# 
#     - $\sqrt{252}$ is the square root of the number of trading days in a year (typically used for daily price data) 
#     - Standard Deviation is the measure of dispersion calculated from the logarithmic returns of the financial instrument over the specified period.
#     - This formula is often applied to historical price data to estimate the volatility of a financial instrument. It provides a standardized measure of the instrument's price fluctuations, allowing investors to compare the volatility of different assets and make informed investment decisions.
# 
# - **Application:** 
# 
#     Volatility is a measure of the degree of variation or dispersion in the price or returns of a financial instrument. A higher volatility indicates larger price swings, while lower volatility suggests relatively stable prices.
# 
#     Based on the volatility index, here are a few things that can be inferred or predicted:
# 
#     - **Market Risk:** Higher volatility generally implies higher market risk. It suggests that prices can fluctuate significantly in a short period, indicating a higher level of uncertainty and potential for larger losses or gains.
# 
#     - **Trading Opportunities:** Volatility can present trading opportunities for investors and traders. Higher volatility can create more significant price movements, providing potential opportunities for profit through well-timed trades or strategies.
# 
#     - **Option Pricing:** Volatility is a crucial factor in determining option prices. Higher volatility leads to higher option premiums since there is an increased likelihood of larger price swings and potential profits for option holders.
# 
#     - **Investor Sentiment:** Volatility can reflect investor sentiment and market conditions. During periods of high volatility, it often indicates increased fear, uncertainty, or market turbulence. Conversely, low volatility may suggest a more stable and confident market environment.
# 
#     - **Risk Management:** Volatility measures can be used for risk management purposes. Investors and portfolio managers may adjust their strategies, asset allocations, or position sizes based on the level of volatility to manage their exposure to market risk.
# 
#     It is important to note that volatility alone cannot provide a complete assessment of market risk. Other factors such as economic conditions, geopolitical events, and market fundamentals should also be considered in determining the overall risk level.

# %%
# Create a new dataframe with selected columns
adani_df2 = adani_df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()

# Calculate the daily returns
returns = adani_df2['Close'].pct_change()

# Calculate the average return
average_return = returns.mean()

# Calculate the squared differences from the average return
squared_diff = (returns - average_return)**2

# Calculate the sum of squared differences
sum_squared_diff = squared_diff.sum()

# Calculate the volatility (square root of the average squared differences)
volatility = np.sqrt(sum_squared_diff / len(returns))

print("Volatility:", volatility)

# %%
# Calculate the logarithmic returns
log_returns = np.log(adani_df2['Close'] / adani_df2['Close'].shift(1))

# Calculate the volatility (standard deviation of log returns)
volatility = log_returns.std()

print("Volatility:", volatility)


# %%
# Assuming your DataFrame for last one month
start_date = pd.to_datetime('2023-03-28')
end_date = pd.to_datetime('2023-04-28')

last_one_month_data = adani_df1.loc[start_date:end_date]

returns = last_one_month_data['Close'].pct_change()
volatility1 = returns.std()
volatility1

# %%
# Calculate the returns
returns = adani_df['Close'].pct_change()

# Calculate the volatility
volatility = returns.std()

# Plot the histogram of returns
plt.hist(returns, bins=30, alpha=0.5, color='blue', edgecolor='black', label='Returns')

# Add a vertical line to indicate the volatility
plt.axvline(x=volatility, color='red', linestyle='--', linewidth=2, label='Volatility')

# Set plot labels and title
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.title('Histogram of Returns with Volatility')

# Add legend
plt.legend()

# Show the plot
plt.show()

# %% [markdown]
# ## 5. Normalization
# 
# - Normalization of data in a DataFrame is the process of scaling the values to a specific range or distribution. 
# - It is commonly done to bring all the variables to a similar scale and remove any bias or distortion caused by differences in the magnitude of values.
# - The use of normalization in a DataFrame has several benefits:
# 
#     - **Improved Comparability:** By scaling the variables to a common range, normalization allows for a fairer and more meaningful comparison between different variables. It ensures that no single variable dominates the analysis simply due to its larger values.
# 
#     - **Elimination of Skewness:** Normalization helps in removing the skewness or imbalance in the distribution of data. It makes the data distribution more symmetrical and reduces the impact of outliers.
# 
#     - **Effective Model Training:** Many machine learning algorithms, such as gradient descent-based methods, perform better when the input features are on a similar scale. Normalizing the data can help in improving the convergence of these algorithms and prevent any one variable from dominating the model training process.
# 
#     - **Interpretability:** Normalization makes it easier to interpret the impact or importance of each variable. When variables are on the same scale, it becomes simpler to compare their coefficients or weights in a model and understand their relative influence.
# 
#     - **Data Visualization:** Normalized data can enhance data visualization by preventing one variable from overshadowing others due to its larger magnitude. It allows for a more accurate representation of the relationships and patterns among variables.
# 
# - The mathematical formula used for Min-Max normalization is:
#   
#     $\boxed{X_{\rm scaled} = \frac{X - X_{\rm min}}{X_{\rm max} - X_{\rm min}}}$
# 
#     where:
# 
#     - $X$ is the original value of a data point.
#     - $X_{\rm scaled}$ is the normalized value of the data point.
#     - $X_{\rm min}$ is the minimum value of the variable being normalized.
#     - $X_{\rm max}$ is the maximum value of the variable being normalized.
# 
#     This formula scales the values of a variable to the range $[0, 1]$. The minimum value $X_{\rm min}$ is transformed to $0$, the maximum value $X_{\rm max}$ is transformed to $1$, and all other values are scaled proportionally in between.

# %%
from sklearn.preprocessing import MinMaxScaler

# Droping previosuly created columns 'Daily Change' and '% Daily Change' columns from adani_df
adani_df_dropped = adani_df.drop(['Daily Change', '% Daily Change'], axis=1)

# Make a copy of the original DataFrame
adani_df_dropped_copy = adani_df_dropped.copy()

# Create a copy of the DataFrame to preserve the original data
adani_df_normalized = adani_df_dropped_copy.copy()

# Select the columns to normalize
columns_to_normalize = ['Open', 'High', 'Low', 'Close', 'Volume']

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Normalize the selected columns
adani_df_normalized[columns_to_normalize] = scaler.fit_transform(adani_df_dropped_copy[columns_to_normalize])

''' 
fit_transform() method to perform the Min-Max scaling on the selected columns of adani_df. 
The fit_transform() method scales the values in each column to the range [0, 1].
'''

# %%
# Display the normalized DataFrame
adani_df_normalized.head()

# %%
# If date is not datetime format then Convert the 'Date' column to datetime type using
# adani_df['Date'] = pd.to_datetime(adani_df['Date'])

# Set the figure size
plt.figure(figsize=(10, 6))
# Plot 'Open' price
sns.lineplot(x='Date', y='Open', data=adani_df_normalized, label='Open')

# Plot 'Close' price
sns.lineplot(x='Date', y='Close', data=adani_df_normalized, label='Close')

# Plot 'High' price
sns.lineplot(x='Date', y='High', data=adani_df_normalized, label='High')

# Plot 'Low' price
sns.lineplot(x='Date', y='Low', data=adani_df_normalized, label='Low')

# Set the labels and title
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Normalized Adani Share Data')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Display the legend
plt.legend()

# Show the plot
plt.show()

# %%
adani_df_normalized.head()

# %%
# Scatter Plot with Marginal Histograms along with linear regression
sns.jointplot(data=adani_df_normalized.loc['2020-01-01':], x='High', y='Volume', kind="reg", color='blue', marker=".")

# %% [markdown]
# ## 6. Rolling statistics 
# 
# - **Rolling mean or rolling average:**
#   
#     $\text{Rolling Mean} = \frac{\text{Sum of values in the window}}{\text{Number of values in the window}}$
# 
# - **Rolling standard deviation:**
#   
#     $\text{Rolling Standard Deviation} = \sqrt{\frac{\text{Sum of squared differences from the mean in the window}}{\text{Number of values in the window}}}$
# 
#     - In both formulas, the "window" refers to the specified number of consecutive rows to include in the calculation. 
#     - The window size determines the number of values over which the rolling mean and rolling standard deviation are calculated.

# %% [markdown]
# **Example:**
# 
# 1. **Rolling mean:** The rolling mean over a window of 10 days for the 'Close' column can be calculated as follows:
# 
# - Let's assume the 'Close' values are denoted as $C_0, C_1, C_2, ..., C_n$ for the corresponding dates $D_1, D_2, D_3, ..., D_n$.
# - To calculate the rolling mean at each date $D_i$ (where $i \geq 9$), you take the average of the 'Close' values for the previous 10 days, i.e.:
# 
#     $\boxed{\text{Rolling Mean (RM) at} D_i = \frac{C_i + C_{i-1} + C_{i-2} + ... + C_{i-9}}{10}}$
# 
# - In other words, you sum up the 'Close' values for the current date and the previous 9 dates and then divide the sum by 10 to get the rolling mean at that date.
# 
# - For example, for $D_9$, the rolling mean will be:
# 
#     $\text{RM at} D_9 = \frac{C_9 + C_8 + C_7 + ... + C_0}{10}$.
# 
#     And for $D_{10}$, the rolling mean will be:
# 
#     $\text{RM at} D_{10} = \frac{C_{10} + C_9 + C_8 + ... + C_1}{10}$
# 
#     You can use this formula to calculate the rolling mean for all the dates from $D_9$ to $D_n$.
# 
# 2. **Rolling standard deviation:** To calculate the rolling standard deviation over a window of 10 days for the 'Close' column, you can follow these steps:
# 
#     $\text{Rolling Standard Deviation (RSD) at} D_i = \sqrt{\frac{(C_i - RM_i)^2 + (C_{i-1} - RM_i)^2 + ... + (C_{i-9} - RM_i)^2}{10}}$
# 
#     In this formula, $RM_i$ represents the rolling mean at date $D_i$, and $C_i$, $C_{i-1}$, ..., $C_{i-9}$ represent the 'Close' values for the current date and the previous 9 dates within the window.

# %%
adani_df3 = adani_df0.copy()

# Calculate the rolling mean
rolling_mean = adani_df3['Close'].rolling(window=10).mean()

# Calculate the rolling standard deviation
rolling_std = adani_df3['Close'].rolling(window=10).std()

''' 
In this case, window=20 indicates that the rolling window size is set to 20, 
meaning that the rolling mean is calculated over each group of 20 consecutive values in the 'Close' column.
'''


# %%
adani_df3['rolling_mean'] = rolling_mean
adani_df3['rolling_std'] = rolling_std
adani_df3

# %%
# Plot the rolling mean
plt.figure(figsize=(10, 6))
plt.plot(adani_df3.index, adani_df3['Close'], label = 'Close', color = 'green')
plt.plot(adani_df3.index, adani_df3['rolling_mean'], label = 'Rolling mean', color='blue', linestyle=':')
plt.plot(adani_df3.index, adani_df3['rolling_std'], label = 'Rolling standard deviation', color='red', linestyle='--')
plt.title('Rolling Mean & rolling std of Close Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# ## 7. Momentum
# 
# - Momentum is a technical indicator used to measure the rate of change in a stock's price over a specific period. It helps identify the strength or weakness of a price trend and is commonly used to confirm trends and generate trading signals.
# 
# - The formula to calculate momentum is quite simple:
# 
#     $\text{Momentum} = \text{Close Price} - \text{Close Price}(n-\text{periods ago})$
# 
#     Where:
# 
#     - Close Price is the current closing price of the stock.
#     - Close Price (n) periods ago is the closing price of the stock n periods (e.g., days) ago.
#     
# - The value of the momentum indicator can be positive or negative. A positive momentum value indicates upward price movement, while a negative momentum value indicates downward price movement.
# 
# - Traders often use a specific lookback period (n) when calculating momentum, such as 14 or 20 days, depending on their trading strategy and timeframe.

# %%
# Calculate the momentum
n_periods = 10  # Number of periods for momentum calculation
adani_df3['Momentum'] = adani_df3['Close'] - adani_df3['Close'].shift(n_periods)

# Print the DataFrame with momentum values
adani_df3.head()

# %%
# Create a figure and axis object
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the Momentum
plt.plot(adani_df3.index, adani_df3['Momentum'], label='Momentum', color='blue')

# Plotting the Close Price
plt.plot(adani_df3.index, adani_df3['Close'], label='Close', color='green')

# Plotting the Moving Average
plt.plot(adani_df3.index, adani_df3['rolling_mean'], label='Moving Average', color='red')

# Add grid lines
plt.grid(True)

# Add a horizontal line at y=0
plt.axhline(y=0, color='black', linestyle='--')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Momentum and Other Lines')
plt.legend()

# Display the plot
plt.show()

# %% [markdown]
# ## 8. Exponential Moving Average
# 
# - Exponential Moving Average (EMA) is a commonly used smoothing technique in time series analysis. 
# - It is similar to the simple moving average (SMA), but it assigns more weightage to recent data points. 
# - The EMA is calculated by applying a smoothing factor to the previous EMA value and adding a fraction of the current value.
# - The formula to calculate the EMA at time period t is:
# 
#     $\text{EMA}_t = (1 - \alpha) \times \text{EMA}_{t-1} + \alpha \times \text{Value}_t$
# 
#     Where:
# 
#     - $\text{EMA}_t$ is the EMA at time period $t$.
#     - $\text{EMA}_{t-1}$ is the EMA at the previous time period $(t-1)$.
#     - $\text{Value}_t$ is the value at time period $t$.
#     - $\alpha$ is the smoothing factor that determines the weightage given to the current value.
#       -  It is usually chosen between $0$ and $1$, smaller values giving more weight to past values and larger values giving more weight to the current value. 
#       -  A commonly used smoothing factor is 2 / (selected period + 1).
# 
# - **Step to follow:**
#   To calculate the Exponential Moving Average (EMA), you can follow these steps:
# 
#   - Choose a period for the EMA, which determines the number of data points to consider for smoothing. For example, you may select a period of 10 days.
#   - Calculate the initial SMA (Simple Moving Average) for the selected period. This can be done by taking the average of the closing prices over the specified period.
#   - Choose a smoothing factor, often denoted as α (alpha), which determines the weight given to the current value versus the previous EMA. The value of α is usually between 0 and 1, with smaller values giving more weight to past values and larger values giving more weight to the current value. A commonly used smoothing factor is 2 / (selected period + 1).
# 
#   - Calculate the EMA for the next data point using the formula:
# 
#     $\text{EMA}_t = (\text{Value}_t - \text{EMA}_{t-1}) \times \alpha + \text{EMA}_{t-1}$
# 
#     where $\text{EMA}_t$ is the EMA at time period $t$, $\text{Value}_t$ is the value at time period $t$, and $\text{EMA}_{t-1}$ is the EMA at the previous time period $(t-1)$.
# 
#   - Repeat step 4 for each subsequent data point, using the previously calculated EMA as the EMA_{t-1} for the next calculation.
# 
# > **Simple Moving Average:** The Simple Moving Average (SMA) is a commonly used method for smoothing time series data and identifying
# > trends. It is calculated by taking the average of a specified number of data points over a given period.
# > 
# > Formula: 
# > 
# >   $\text{SMA} = \frac{\text{Sum of data points for the specified period}}{\text{Number of data points in the specified period}}$

# %%
adani_df4 = adani_df0.copy()

adani_df4.head()

# %%
# Specify the window size for the SMA
window = 20

# Calculate the SMA using rolling mean
adani_df4['SMA'] = adani_df4['Close'].rolling(window).mean()
adani_df4.head()

# %%
# Specify the span for the EMA
span = 20

# Calculate the EMA using ewm
adani_df4['EMA'] = adani_df4['Close'].ewm(span=span, adjust=False).mean()

# Print the DataFrame with EMA

adani_df4.head(20)

# %%
# Create a figure and axis object
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(adani_df4.index, adani_df4['High'], label='High', color='black', linestyle = '--')
ax.plot(adani_df4.index, adani_df4['Close'], label='Close', color='green', linestyle = ':')
# Plot SMA
ax.plot(adani_df4.index, adani_df4['SMA'], label='SMA', color='blue')

# Plot EMA
ax.plot(adani_df4.index, adani_df4['EMA'], label='EMA', color='red')

# Set the x-axis label
ax.set_xlabel('Date')

# Set the y-axis label
ax.set_ylabel('Moving Average')

# Set the title
ax.set_title('SMA and EMA Variation')

# Add a legend
ax.legend()

# Display the plot
plt.show()

# %% [markdown]
# ## 9. Bollinger Bands
# - Bollinger Bands are a popular technical analysis tool used to identify volatility and potential price reversal points in financial markets.
# - The Bollinger Bands consist of three lines:
# 
#     - The middle band, which is typically a `Simple Moving Average (SMA)` over a specified period.
#     - The upper band, which is calculated by adding a specified number of `standard deviations` to the middle band.
#     - The lower band, which is calculated by subtracting a specified number of `standard deviations` from the middle band.
# 
# - The formula to calculate Bollinger Bands is as follows:
# 
#     - Middle Band (MB) = SMA                               (Simple Moving Average)
#     - Upper Band (UB) = MB + (k * Standard Deviation)
#     - Lower Band (LB) =  MB - (k * Standard Deviation)
#     
#     Where:
# 
#     - k is the number of standard deviations to use (typically set to 2)
#     - Standard Deviation is calculated as the square root of the mean of squared differences between each data point and the mean of the specified period.
# 
# - The Bollinger Bands help traders identify periods of high or low volatility and potential price reversals. 
# - When the price touches or crosses the upper band, it may indicate an overbought condition, suggesting a potential price decrease. 
# - Conversely, when the price touches or crosses the lower band, it may indicate an oversold condition, suggesting a potential price increase.

# %%
# Specify the period and the number of standard deviations
period = 20
num_std = 2

# Calculate the middle band (MB) using a Simple Moving Average (SMA)
adani_df4['MB'] = adani_df4['Close'].rolling(window=period).mean()

# Calculate the standard deviation (SD) using the same period
adani_df4['SD'] = adani_df4['Close'].rolling(window=period).std()

# Calculate the upper band (UB) and lower band (LB)
adani_df4['UB'] = adani_df4['MB'] + (num_std * adani_df4['SD'])
adani_df4['LB'] = adani_df4['MB'] - (num_std * adani_df4['SD'])

adani_df4.head()

# %%
# Plot the closing prices
plt.figure(figsize=(10, 6))
plt.plot(adani_df4.index, adani_df4['Close'], label='Close', color='blue')

# Plot the middle band (MB)
plt.plot(adani_df4.index, adani_df4['MB'], label='Middle Band', color='black', linestyle = '--')

# Plot the upper band (UB)
plt.plot(adani_df4.index, adani_df4['UB'], label='Upper Band', color='red', linestyle = ':')

# Plot the lower band (LB)
plt.plot(adani_df4.index, adani_df4['LB'], label='Lower Band', color='green', linestyle = ':')


# Shade the area between UB and LB
plt.fill_between(adani_df4.index, adani_df4['UB'], adani_df4['LB'], color='lightgray', alpha=0.3)


# Add labels and title
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Bollinger Bands')
plt.grid(True)

# Add legend
plt.legend()

# Display the plot
plt.show()


# %% [markdown]
# ## 10. Moving average convergence divergence (MACD)
# 
# - The Moving Average Convergence Divergence (MACD) is a popular technical indicator used in financial analysis to identify potential trend reversals, generate buy and sell signals, and assess the overall momentum of an asset's price. The MACD is calculated using the following formula:
# 
#     - MACD Line (MACD): EMA(12-period) - EMA(26-period)
#     - Signal Line: EMA(9-period) of the MACD Line
#     - MACD Histogram: MACD Line - Signal Line   $\Rightarrow$ It signifies the strength and momentum of the price movement.
# 
#     Here's a breakdown of the components:
# 
#     - MACD Line (MACD): It is the difference between the 12-period Exponential Moving Average (EMA) and the 26-period EMA. It captures the short-term and long-term trend of the price.
#     - Signal Line: It is the 9-period EMA of the MACD Line. The Signal Line acts as a trigger line and helps identify potential buy and sell signals.
#     - MACD Histogram: It represents the difference between the MACD Line and the Signal Line. The histogram visually shows the convergence and divergence of the MACD Line and the Signal Line, indicating the strength of the price momentum. 

# %%
# Copying the original dataframe

adani_df5 = adani_df0.copy()

# Calculate MACD
short_period = 12
long_period = 26
signal_period = 9

# Calculate the short-term EMA
ema_short = adani_df5['Close'].ewm(span=short_period, adjust=False).mean()

# Calculate the long-term EMA
ema_long = adani_df5['Close'].ewm(span=long_period, adjust=False).mean()

# Calculate MACD Line
macd_line = ema_short - ema_long

# Calculate Signal Line
signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

# MACD histogram line
macd_histogram = macd_line - signal_line


# Add MACD Line and Signal Line to DataFrame
adani_df5['MACD Line'] = macd_line
adani_df5['Signal Line'] = signal_line
adani_df5['MACD Histogram'] = macd_histogram

adani_df5.head()

# %%
# Plot MACD Line, Signal Line, and Histogram
plt.figure(figsize=(10, 6))
plt.plot(adani_df5.index, adani_df5['MACD Line'], label='MACD Line', color='blue')
plt.plot(adani_df5.index, adani_df5['Signal Line'], label='Signal Line', color='red')
plt.bar(adani_df5.index, adani_df5['MACD Histogram'], label='MACD Histogram', color=np.where(adani_df5['MACD Histogram'] > 0, 'green', 'red'))

# Add zero line
plt.axhline(y=0, color='black', linestyle='--')

# Customize the plot
plt.title('MACD')
plt.xlabel('Date')
plt.ylabel('MACD')
plt.legend()

# Show the plot
plt.show()

# %%
# We can draw MACD line along with the CLosing price along with the Bollinger bands.
plt.figure(figsize=(12, 6))
plt.plot(adani_df5.index, adani_df5['Close'], label='Adani Close', color='black')
plt.plot(adani_df5.index, adani_df5['MACD Line'], label='MACD Line', color='green')
plt.plot(adani_df5.index, adani_df5['Signal Line'], label='Signal Line', color='blue')
plt.bar(adani_df5.index, adani_df5['MACD Histogram'], color=np.where(adani_df5['MACD Histogram'] > 0, 'green', 'red'))

# Plot the middle band (MB)
plt.plot(adani_df4.index, adani_df4['MB'], label='Middle Band', color='black', linestyle = '--')
# Plot the upper band (UB)
plt.plot(adani_df4.index, adani_df4['UB'], label='Upper Band', color='red', linestyle = ':')
# Plot the lower band (LB)
plt.plot(adani_df4.index, adani_df4['LB'], label='Lower Band', color='green', linestyle = ':')
# Shade the area between UB and LB
plt.fill_between(adani_df4.index, adani_df4['UB'], adani_df4['LB'], color='lightgray', alpha=0.3)


plt.axhline(y=0, color='black', linewidth=0.5)
plt.title('MACD and Adani Share Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# %% [markdown]
# ## 11. Stochastic (STO) & Relative Strength Index (RSI)
# 
# Stochastic (STO) and Relative Strength Index (RSI) are popular technical indicators used in financial analysis to evaluate the momentum and strength of price movements in a security. 
# 
# ### 1. Stochastic (STO)
# 
# - The Stochastic oscillator measures the closing price relative to the high-low range over a specified period.
# 
#     $\% \text{K} = \left(\frac{\text{Current Close} - \text{Lowest Low}}{\text{Highest High} - \text{Lowest Low}}\right) \times 100$
# 
#     $\%\text{D} = \text{Simple moving average of} ~\%\text{K}~ \text{over a specified period}$
# 
#     Where:
# 
#     - Current Close: The most recent closing price.
#     - Lowest Low: The lowest low price over a specified period.
#     - Highest High: The highest high price over a specified period.
#     - %K: The current value of the Stochastic oscillator.
#     - %D: The moving average of %K over a specified period.
# 
# ### 2. Relative Strength Index (RSI):
# - The Relative Strength Index measures the magnitude of recent price changes to assess overbought or oversold conditions in a security.
# 
#     $\text{RSI} = 100 - \frac{100}{1 + RS}$
#     
#     $\text{RS} = \frac{\text{Average gain over a specified period}}{\text{Average loss over a specified period}}$
# 
#     Where:
# 
#     - Average gain: The average price increase over a specified period.
#     - Average loss: The average price decrease over a specified period.
#     - RS: The relative strength calculated as the ratio of the average gain to the average loss.
#     - RSI: The current value of the Relative Strength Index.
# 
# > **Note:** The specific period and parameters used in the calculations can vary based on individual preferences and trading strategies.
# >
# > Please note that these formulas provide a basic understanding of how STO and RSI are calculated. There might be variations or different smoothing techniques used in different implementations.

# %%
# copying the original dataframe

adani_df6 = adani_df0.copy()

# %% [markdown]
# We can calculate STO as:
# ```
# # Calculate Stochastic Oscillator (STO)
# def calculate_stochastic(data, k_period=14, d_period=3):
#     highest_high = data['High'].rolling(window=k_period).max()
#     lowest_low = data['Low'].rolling(window=k_period).min()
#     k = ((data['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
#     d = k.rolling(window=d_period).mean()
#     return k, d
# 
# ```

# %% [markdown]
# We can calculate RSI as:
# ```
# # Calculate RSI
# def calculate_rsi(data, period=14):
#     close_delta = data['Close'].diff()
#     gain = close_delta.where(close_delta > 0, 0)
#     loss = -close_delta.where(close_delta < 0, 0)
#     avg_gain = gain.rolling(window=period).mean()
#     avg_loss = loss.rolling(window=period).mean()
#     rs = avg_gain / avg_loss
#     rsi = 100 - (100 / (1 + rs))
#     return rsi
# ```

# %%
highest_high = adani_df6['High'].rolling(14).max()
lowest_low = adani_df6['Low'].rolling(14).min()
adani_df6['%K'] = (adani_df6['Close']-lowest_low)*100/(highest_high-lowest_low)
adani_df6['%D'] = adani_df6['%K'].rolling(3).mean() # check the rolling window,

# %%
adani_df6.head()

# %%
plt.figure(figsize=(10, 6))
plt.plot(adani_df6.index, adani_df6['RSI'])
plt.title('RSI')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.grid(True)
plt.show()


# %%
plt.figure(figsize=(10, 6))
plt.plot(adani_df6.index, adani_df6['%K'], label='%K')
plt.plot(adani_df6.index, adani_df6['%D'], label='%D')
plt.title('Stochastic Oscillator (%K and %D)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()


# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(adani_df6.index, adani_df6['RSI'])
ax1.set_title('RSI')
ax1.set_xlabel('Date')
ax1.set_ylabel('RSI')
ax1.grid(True)

ax2.plot(adani_df6.index, adani_df6['%K'], label='%K')
ax2.plot(adani_df6.index, adani_df6['%D'], label='%D')
ax2.set_title('Stochastic Oscillator (%K and %D)')
ax2.set_xlabel('Date')
ax2.set_ylabel('Value')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()


# %%
# Create a figure with two subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Plot Close in the second subplot
ax1.plot(adani_df6.index, adani_df6['Close'], label='Close')
ax1.set_xlabel('Date')
ax1.set_ylabel('Close')
ax1.set_title('Close')
ax1.legend()
ax1.grid(True)

# Plot RSI, %K, and %D in the first subplot
ax2.plot(adani_df6.index, adani_df6['RSI'], label='RSI', color = 'red')
ax2.grid(True)
ax2.set_ylabel('Value')
ax2.legend()

ax3.plot(adani_df6.index, adani_df6['%K'], label='%K', color = 'blue', linestyle =  ':')
ax3.plot(adani_df6.index, adani_df6['%D'], label='%D', color = 'green', linestyle = ':')
ax3.set_ylabel('Value')
ax3.legend()
ax3.grid(True)

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()


# %% [markdown]
# ## Reference
# 
# - https://github.com/arunsinp/Data-analysis-projects
# - https://github.com/arunsinp/Data-analysis-projects/tree/main/EDA_dbbank
# - https://github.com/arunsinp/Data-analysis-projects/blob/main/EDA_dbbank/Statistics.ipynb


