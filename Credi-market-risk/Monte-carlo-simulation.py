# %% [markdown]
# # Monte Carlo Simulation: History, How it Works, and 4 Key Steps
# 
# Monte Carlo simulation is a computational technique used to model and analyze complex systems or processes by generating random samples and simulating their outcomes. It is particularly useful in situations where there is uncertainty and variability involved.
# 
# ### Simulation Process
# Here's an overview of the Monte Carlo simulation process:
# 
# - **Define the Problem:** Clearly define the problem or system you want to model. Identify the variables and parameters that have an impact on the outcomes of interest.
# 
# - **Define Probability Distributions:** Assign probability distributions to the variables that exhibit uncertainty. These distributions can be based on historical data, expert judgment, or assumptions.
# 
# - **Generate Random Samples:** Randomly generate values for the variables based on their assigned distributions. The number of samples should be large enough to obtain reliable results.
# 
# - **Perform Simulations:** For each set of random values generated, simulate the system or process and calculate the desired outputs or outcomes. This involves using mathematical models, equations, or algorithms that describe the behavior of the system.
# 
# - **Collect Results:** Store the results obtained from each simulation run, such as performance measures, metrics, or output variables of interest.
# 
# - **Analyze Results:** Analyze the collected results to gain insights into the behavior and characteristics of the system. This can involve statistical analysis, data visualization, or other techniques to summarize and interpret the simulation outcomes.
# 
# - **Draw Conclusions:** Draw conclusions based on the analysis of the simulation results. Assess the reliability and robustness of the conclusions by considering the variability and uncertainty inherent in the simulation process.
# 
# ### Application
# 
# Monte Carlo simulation can be applied in various fields, including 
# - finance, 
# - engineering, 
# - risk analysis, 
# - optimization, and 
# - decision-making. 
# 
# It allows for the exploration of different scenarios and the assessment of the likelihood and impact of different outcomes.
# 
# By repeatedly simulating the system with random inputs, Monte Carlo simulation provides a probabilistic understanding of the system's behavior and can help decision-makers make more informed choices, evaluate risks, and assess the potential impact of different strategies or interventions.
# ### Python libraries important for the simulation
# 
# Python provides libraries such as 
# - NumPy and 
# - SciPy 
#   
# that offer functions for generating random numbers, statistical analysis, and mathematical modeling, making it a popular programming language for implementing Monte Carlo simulations.

# %% [markdown]
# ## Monte Carlo Simulation in Finance
# 
# Monte Carlo simulation is widely used in finance for various purposes, for example:
# 
# - **Risk Analysis:** Monte Carlo simulation is used to assess and quantify financial risk. By simulating various market scenarios and generating multiple outcomes, it helps estimate the potential range of portfolio returns or losses. It allows analysts to evaluate the likelihood of different risk levels, such as Value-at-Risk (VaR) or Expected Shortfall (ES), and understand the distribution of possible outcomes.
# 
# - **Portfolio Optimization:** Monte Carlo simulation can be used to optimize investment portfolios by considering various asset allocation strategies. By simulating the performance of different portfolio compositions and rebalancing methods, it helps investors determine the optimal allocation that maximizes returns while managing risk within specified constraints.
# 
# - **Option Pricing:** Monte Carlo simulation is employed in option pricing models, such as the Black-Scholes model, to estimate the value of options and other derivative securities. By simulating the underlying asset price paths, it helps calculate the probability distribution of option payoffs and determine fair prices for options.
# 
# - **Credit Risk Assessment:** Monte Carlo simulation is used to assess credit risk by simulating the potential outcomes of loans or credit portfolios. It helps estimate the probability of default, loss given default, and other credit-related metrics. By incorporating factors such as default correlations, recovery rates, and macroeconomic variables, it enables the modeling of credit portfolio risks and the calculation of credit value adjustments (CVAs).
# 
# - **Financial Forecasting and Scenario Analysis:** Monte Carlo simulation is employed in financial modeling to generate forecasts and assess the impact of different scenarios on financial performance. It allows analysts to incorporate uncertainty and variability in key inputs such as sales growth rates, interest rates, or exchange rates. By simulating multiple scenarios, it provides insights into the range of possible outcomes and helps inform decision-making.
# 
# - **Stress Testing:** Monte Carlo simulation is utilized in stress testing exercises to assess the resilience of financial institutions or portfolios under extreme market conditions. It helps evaluate the impact of severe shocks on capital adequacy, liquidity, and other risk metrics. By simulating a wide range of adverse scenarios, it assists in identifying vulnerabilities and designing risk mitigation strategies.

# %% [markdown]
# #### Covariance matrix:
# 
# - The covariance matrix, is a square matrix that represents the covariance between multiple variables. 
# - In the context of finance, it is commonly used to analyze the relationships between the returns of different assets in a portfolio. 
# - The formula to calculate the covariance between two variables, X and Y, is as follows:
# 
#     $Cov(X, Y) = E[(X - \mu_X) * (Y - \mu_Y)]$
# 
#     where:
# 
#     - $Cov(X, Y)$ is the covariance between variables X and Y.
#     - $E[ ]$ denotes the expectation or average value.
#     - $X$ and $Y$ are random variables or data series.
#     - $\mu_X$ and $\mu_Y$ are the means (or expected values) of X and Y, respectively.
# 
# - The covariance matrix is formed by calculating the covariances between all pairs of variables in a dataset. 
# - For a dataset with $n$ variables, the covariance matrix $\Sigma$ will be an $n \times n$ matrix, where each element $\Sigma_{i,j}$ represents the covariance between the ith and jth variables.
# - The general formula to calculate the covariance matrix is:
# 
# $$
# \text{cov}(X,Y) = 
# \begin{bmatrix}
# \text{Cov}(X_1, X_1) & \text{Cov}(X_1, X_2) & \ldots & \text{Cov}(X_1, X_n) \\
# \text{Cov}(X_2, X_1) & \text{Cov}(X_2, X_2) & \ldots & \text{Cov}(X_2, X_n) \\
# \vdots & \vdots & \ddots & \vdots \\
# \text{Cov}(X_n, X_1) & \text{Cov}(X_n, X_2) & \ldots & \text{Cov}(X_n, X_n)
# \end{bmatrix}
# $$
# 
# 
# 
# where: $Cov(X_i, X_j)$ represents the covariance between the ith and jth variables.
# 
# - In Python, we can use the `.cov()` function from the NumPy or pandas library to compute the covariance matrix from a given dataset.
# 
# - `covMatrix = returns.cov()`: This line calculates the covariance matrix of the returns using the `cov()` method of the returns DataFrame. The covariance matrix provides insights into the relationship between the returns of different stocks and is used in portfolio optimization.

# %% [markdown]
# #### Expected portfolio return:
# 
# - The expected portfolio return is a measure of the anticipated average return that an investor can expect from a portfolio of investments. It takes into account the weights assigned to each asset in the portfolio and their respective expected returns.
# - We can calculate it in two ways. 
#   - **First way:**
# 
#     $\text{Expected Portfolio Return} = \sum_i (\text{Expected Return}_i \times \text{Weight}_i)$
# 
#     where:
# 
#     - Expected Portfolio Return represents the weighted average of the expected returns of individual assets in the portfolio.
#     - Expected Return_i represents the expected return of asset i.
#     - Weight_i represents the weight assigned to asset i in the portfolio.
# 
#     This formula calculates the expected return of a portfolio by summing the products of the expected return of each asset (Expected Return_i) and its corresponding weight (Weight_i) in the portfolio. It assumes that the expected returns provided for each asset are directly representative of the future performance of the assets.
# 
#   - **Second way:**
#     
#     $\text{Expected Portfolio Return} = \sum_i (\text{Mean Return}_i \times \text{Weight}_i)\times \text{time}$
# 
#       - This formula calculates the expected return of a portfolio by summing the products of the mean return of each asset (Mean Return_i) and its corresponding weight (Weight_i) in the portfolio. 
#       - However, it also takes into account the time period (Time) for which the expected return is calculated. 
#       - By multiplying the sum of the weighted mean returns by the time period, it provides an estimate of the expected return over the specified time frame.
# 
# - The expected portfolio return provides an estimate of the average return that an investor can expect from their portfolio based on the anticipated performance of the individual assets and their respective weightings. It is a key metric used in portfolio analysis and decision-making.

# %% [markdown]
# #### 1. Var & CVaR calculation

# %%
%pip install pandas_datareader -q

# %% [markdown]
# ### Standard deviation and Voltality
# 
# - The formula to calculate the standard deviation of a portfolio is as follows:
# 
#     $\text{std} = \sqrt{\text{Transpose}(w) \times \Sigma  \times w} \times \sqrt(t)$
# 
#     where:
# 
#     - std represents the standard deviation of the portfolio.
#     - $w$ is a vector of weights assigned to each asset in the portfolio.
#     - $\Sigma$ is the covariance matrix of asset returns.
#     - T is the time period over which the portfolio returns are calculated.
# 
# 
# 
# - standard deviation is defined as:
# 
#     $\boxed{\text{Volatility} = \sqrt{\text{Variance}} = \sqrt{t} \times \text{Standard deviation}}$
# 
#     here 252 is for number days of trading in a year t =252
# 
# - **Example:**
# 
#     Suppose we have a portfolio with three assets, and we want to calculate the standard deviation of the portfolio. We have the following information:
# 
#     Weights:    w = [0.4, 0.3, 0.3]
# 
#     Covariance Matrix:
#         Σ = [[0.04, 0.03, 0.02],
#         [0.03, 0.09, 0.05],
#         [0.02, 0.05, 0.16]]
# 
#     To calculate the standard deviation, we can use the formula:
# 
#     σ = sqrt( w^T * Σ * w )
# 
#     First, let's calculate w^T * Σ:
# 
#     w^T * Σ = [0.4, 0.3, 0.3] * [[0.04, 0.03, 0.02],
#                                 [0.03, 0.09, 0.05],
#                                 [0.02, 0.05, 0.16]]
#                                 
#         = [0.4*0.04 + 0.3*0.03 + 0.3*0.02, 0.4*0.03 + 0.3*0.09 + 0.3*0.05, 0.4*0.02 + 0.3*0.05 + 0.3*0.16]
# 
#         = [0.033, 0.048, 0.041]
# 
#     w^T * Σ * w = [0.033, 0.048, 0.041] * [0.4, 0.3, 0.3] = 0.0176
# 
#     Therefore
# 
#     σ = sqrt(0.0176) = 0.1326
# 
#     So, the standard deviation of the portfolio in this example is approximately 0.1326. This represents the measure of risk or volatility associated with the portfolio based on the given weights and covariance matrix.

# %% [markdown]
# ##### 1. Portfolio performance analysis

# %% [markdown]
# **Step-1:** Importing Libraries
# 
# Import the required libraries in your Python script:

# %%
# Import Libraries (Import the required libraries in your Python script)
import pandas as pd
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf

# %% [markdown]
# **Step-2:** Define Function to Get Stock Data
# 
# Define a function that retrieves stock data from Yahoo Finance for a given list of symbols, start date, and end date:

# %%
def getData(stocks, start, end):
    stockData = pdr.get_data_yahoo([stock + '.AX' for stock in stocks], start=start, end=end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return returns, meanReturns, covMatrix

# %% [markdown]
# Portfolio performace

# %%
def portfolioPerformance(weights, meanReturns, covMatrix, Time):
    portfolioReturn = np.sum(meanReturns * weights) * Time
    portfolioStd = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(Time)
    return portfolioReturn, portfolioStd

# %% [markdown]
# **Step 3:** Specify Stock Symbols and Date Range
# 
# Specify the stock symbols and the desired date range for which you want to retrieve the data:

# %%
stocklist = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=800)

# %% [markdown]
# **Step 4:** Retrieve Stock Data
# 
# Call the get_stock_data function to retrieve the stock data for the specified symbols and date range:

# %%
returns, meanReturns, covMatrix = getData(stocklist, start=startDate, end=endDate)

# %% [markdown]
# **Step 6:** Drop any rows with missing values (NaN) from the returns dataframe:

# %%
returns = returns.dropna()

# %% [markdown]
# **Step 6:** Define the weights for the portfolio:

# %%
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])  # Example weights, adjust as needed

# %% [markdown]
# Define the investment time period:

# %%
Time = 10  # Example time period, adjust as needed

# %% [markdown]
# **Step 7:** Calculate the portfolio performance:

# %%
portfolioReturn, portfolioStd = portfolioPerformance(weights, meanReturns, covMatrix, Time)

# %%
print("Portfolio Return:", portfolioReturn)
print("Portfolio Standard Deviation:", portfolioStd)

# %%
returns.head()

# %% [markdown]
# Now adding daily Portfolio returns to this table, using following formula:
# 
# 
# Portfolio Returns = Returns of Asset 1 * Weight of Asset 1 + Returns of Asset 2 * Weight of Asset 2 + ... + Returns of Asset N * Weight of Asset N

# %%
returns['Portfolio'] = returns.dot(weights)

# %%
returns.head()

# %% [markdown]
# **Historical VaR:**
# 
# Sort the portfolio returns (returns['portfolio']) in ascending order.
# Determine the confidence level (confidence_level) at which you want to calculate VaR.
# Find the index position (index) that corresponds to the confidence level.
# Calculate the Historical VaR using the formula: historical_var = -returns['portfolio'].iloc[index] * portfolio_value
# Here's an example code snippet for calculating Historical VaR:

# %%
confidence_level = 0.05  # 95% confidence level
portfolio_returns_sorted = returns.sort_values(by='Portfolio')
index = int(confidence_level * len(portfolio_returns_sorted))
var_hist = -portfolio_returns_sorted['Portfolio'].iloc[index]
var_hist

# %%
# Historical VaR for each stock in the portfolio

confidence_level = 0.95  # 95% confidence level
portfolio_value = 100000  # Example portfolio value

# Iterate over each stock in the portfolio
for stock in stocklist:
    stock_returns_sorted = returns[stock + '.AX'].sort_values()
    index = int(confidence_level * len(stock_returns_sorted))
    historical_var = -stock_returns_sorted.iloc[index] * portfolio_value
    print(f"Historical VaR for {stock}: {historical_var}")

# %% [markdown]
# **Conditional Value at Risk (CVaR)**, also known as Expected Shortfall (ES), is a risk measure that represents the expected loss beyond a certain confidence level. It provides information about the average magnitude of losses that exceed the Value at Risk (VaR) estimate.
# 
# To calculate the Conditional VaR, you can follow these steps:
# 
# Set the confidence level (e.g., 95%).
# Calculate the VaR at the specified confidence level.
# Determine the subset of returns that fall below the VaR.
# Calculate the average of the returns in the subset obtained in step 3. This represents the Conditional VaR.
# Here's an example of how you can calculate Conditional VaR for a portfolio:

# %%
confidence_level = 0.95  # 95% confidence level

# Calculate the VaR at the specified confidence level
portfolio_returns_sorted = returns['Portfolio'].sort_values()
index = int(confidence_level * len(portfolio_returns_sorted))
var = -portfolio_returns_sorted[index]

# Determine the subset of returns that fall below the VaR
subset_returns = portfolio_returns_sorted[:index]

# Calculate the Conditional VaR (average of the returns in the subset)
cvar = -np.mean(subset_returns)

print(f"Conditional VaR at {confidence_level}: {cvar}")


# %%
import matplotlib.pyplot as plt

# Calculate VaR and CVaR
confidence_level = 0.95
portfolio_returns_sorted = returns['Portfolio'].sort_values()
index = int(confidence_level * len(portfolio_returns_sorted))
var = -portfolio_returns_sorted[index]
subset_returns = portfolio_returns_sorted[:index]
cvar = -np.mean(subset_returns)

# Plot histogram of portfolio returns
plt.figure(figsize=(10, 6))
plt.hist(portfolio_returns_sorted, bins=50, color='skyblue', edgecolor='black')
plt.axvline(var, color='red', linestyle='dashed', linewidth=2, label='VaR at 95%')
plt.axvline(cvar, color='orange', linestyle='dashed', linewidth=2, label='CVaR at 95%')
plt.xlabel('Portfolio Return')
plt.ylabel('Frequency')
plt.title('Distribution of Portfolio Returns')
plt.legend()
plt.grid(True)
plt.show()

# %%
import matplotlib.pyplot as plt

# Calculate VaR and CVaR
confidence_level = 0.95
portfolio_returns_sorted = returns['Portfolio'].sort_values()
index = int(confidence_level * len(portfolio_returns_sorted))
var = -portfolio_returns_sorted[index]
subset_returns = portfolio_returns_sorted[:index]
cvar = -np.mean(subset_returns)

# Plot histogram of portfolio returns
plt.figure(figsize=(10, 6))
plt.hist(portfolio_returns_sorted, bins=50, color='skyblue', edgecolor='black')
plt.axvline(var, color='red', linestyle='dashed', linewidth=2, label='VaR at 95%')
plt.axvline(cvar, color='orange', linestyle='dashed', linewidth=2, label='CVaR at 95%')

# Plot Historical VaR for each stock
for stock in stocklist:
    stock_returns_sorted = returns[stock + '.AX'].sort_values()
    index = int(confidence_level * len(stock_returns_sorted))
    historical_var = -stock_returns_sorted.iloc[index] * portfolio_value
    plt.axvline(historical_var, color='green', linestyle='dashed', linewidth=2, label=f'Historical VaR for {stock}')

plt.xlabel('Portfolio Return')
plt.ylabel('Frequency')
plt.title('Distribution of Portfolio Returns')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# ## Reference
# 
# - https://www.investopedia.com/terms/m/montecarlosimulation.asp#:~:text=A%20Monte%20Carlo%20simulation%20is%20a%20model%20used%20to%20predict,in%20prediction%20and%20forecasting%20models.


