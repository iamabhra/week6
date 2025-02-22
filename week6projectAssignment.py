import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import stemgraphic
import seaborn as sns
from scipy.special import comb  # For combinatorial calculations (see line 209)
from scipy.stats import hypergeom # For dhyper() equivalent (see line 225)
from scipy.stats import binom # For dbinom() equivalent (see line 264)
from scipy.stats import poisson # For dpois() equivalent (see line 304)
from scipy.stats import nbinom # For dnbinom() equivalent (see line 324)
from scipy.stats import geom # For dgeom() equivalent (see line 338)
from scipy.stats import norm # For pnorm() equivalent (see line 349)
from scipy.stats import probplot # For qqnorm() equivalent (see line 367)
from scipy.stats import lognorm # For plnorm() equivalent (see line 429)
from scipy.stats import expon # For pexp() equivalent (see line 438)
from scipy.stats import gamma # For pgamma() equivalent (see line 453)
from scipy.stats import weibull_min # For pweibull() equivalent (see line 462)
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp

data = pd.read_csv('/Users/abhijitghosh/Documents/DataScience/IN_chemistry.csv')
model1 = smf.ols('Secchi ~ NO3_epi + NH3_epi + Total_Phos_epi', data=data).fit()

# Summary of the model
print(model1.summary())

# Linear regression: Chlorophyll_a ~ Total_Phos_epi + TKN_epi + Secchi
model2 = smf.ols('Chlorophyll_a ~ Total_Phos_epi + TKN_epi + Secchi', data=data).fit()

# Summary of the model
print(model2.summary())

# Check suitability thresholds
criteria = {
    "NO3_epi": 10,  # mg/L
    "NH3_epi": 0.5,  # mg/L
    "Secchi": 1.5    # m
}

import numpy as np
from scipy.stats import ttest_1samp

# Example data (replace with actual lake data from your dataset)
nitrate_levels = data['NO3_epi']  # Example nitrate data (mg/L)
ammonia_levels = data['NH3_epi']    # Example ammonia data (mg/L)
secchi_depths = data['Secchi']     # Example Secchi depth data (m)

# Thresholds for suitability
nitrate_threshold = 10.0
ammonia_threshold = 0.5
secchi_threshold = 1.5

# One-sample t-tests
# Nitrate
nitrate_t_stat, nitrate_p_value = ttest_1samp(nitrate_levels, nitrate_threshold)
print("Nitrate:")
print(f"T-Statistic: {nitrate_t_stat}, P-Value: {nitrate_p_value}")
if nitrate_p_value < 0.05:
    print("Reject the null hypothesis: Nitrate levels exceed the threshold.")
else:
    print("Fail to reject the null hypothesis: Nitrate levels meet the threshold.")

# Ammonia
ammonia_t_stat, ammonia_p_value = ttest_1samp(ammonia_levels, ammonia_threshold)
print("\nAmmonia:")
print(f"T-Statistic: {ammonia_t_stat}, P-Value: {ammonia_p_value}")
if ammonia_p_value < 0.05:
    print("Reject the null hypothesis: Ammonia levels exceed the threshold.")
else:
    print("Fail to reject the null hypothesis: Ammonia levels meet the threshold.")

# Secchi Depth
secchi_t_stat, secchi_p_value = ttest_1samp(secchi_depths, secchi_threshold)
print("\nSecchi Depth:")
print(f"T-Statistic: {secchi_t_stat}, P-Value: {secchi_p_value}")
if secchi_p_value < 0.05:
    print("Reject the null hypothesis: Secchi depth is below the threshold.")
else:
    print("Fail to reject the null hypothesis: Secchi depth meets the threshold.")

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


# Function to plot Q-Q plots
def plot_qq(data, title):
    plt.figure(figsize=(6, 6))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot for {title}")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.grid()
    plt.show()

# Generate Q-Q plots for the three variables
plot_qq(nitrate_levels, "Nitrate Levels")
plot_qq(ammonia_levels, "Ammonia Levels")
plot_qq(secchi_depths, "Secchi Depths")

# Example data (replace with actual dataset)
data = {
    'Lake': ['Lake A', 'Lake B', 'Lake C', 'Lake D', 'Lake E'],
    'Nitrate': [9.2, 11.5, 8.6, 10.2, 7.3],  # mg/L
    'Ammonia': [0.3, 0.6, 0.4, 0.5, 0.2],   # mg/L
    'Secchi': [1.8, 1.4, 2.0, 1.2, 1.7]     # meters
}
df = pd.DataFrame(data)

# Suitability thresholds
nitrate_threshold = 10
ammonia_threshold = 0.5
secchi_threshold = 1.5

# Evaluate suitability
data['Nitrate_Suitable'] = data['NO3_epi'] <= nitrate_threshold
data['Ammonia_Suitable'] = data['NH3_epi'] <= ammonia_threshold
data['Secchi_Suitable'] = data['Secchi'] >= secchi_threshold

# Count the number of lakes meeting each criterion
suitability_counts = {
    'Nitrate': data['Nitrate_Suitable'].sum(),
    'Ammonia': data['Ammonia_Suitable'].sum(),
    'Secchi': data['Secchi_Suitable'].sum()
}

# Convert to percentages
total_lakes = len(data)

suitability_percentages = {key: (value / total_lakes) * 100 for key, value in suitability_counts.items()}

# Plot the bar chart
plt.figure(figsize=(8, 6))
plt.bar(suitability_percentages.keys(), suitability_percentages.values(), color=['blue', 'green', 'orange'])
plt.ylim(0, 100)
plt.title("Percentage of Lakes Meeting Suitability Criteria", fontsize=14)
plt.ylabel("Percentage of Lakes (%)", fontsize=12)
plt.xlabel("Parameters", fontsize=12)
plt.axhline(50, color='red', linestyle='--', label="50% Threshold")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#Assumption

# Re-import necessary libraries after kernel reset
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate synthetic data for illustration (replace with actual dataset)
#Assumption - 2

# Define independent variables (nutrients) and dependent variable (Secchi depth)
X = data[['Total_Phos_epi', 'TKN_epi', 'Secchi']]
y = data['Chlorophyll_a']

# Train a linear regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and calculate residualsy
y_pred = model.predict(X_test)
residuals = y_test - y_pred


# Q-Q plot for residuals
plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)
plt.ylabel("Sample Quantiles", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

#Assumption - 1
X = data[['NO3_epi', 'NH3_epi', 'Total_Phos_epi']]
y = data['Secchi']

# Train a linear regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and calculate residualsy
y_pred = model.predict(X_test)
residuals = y_test - y_pred


# Q-Q plot for residuals
plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)
plt.ylabel("Sample Quantiles", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
