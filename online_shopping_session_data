# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import binom

# ============================================
# LOAD DATA
# ============================================
shopping_data = pd.read_csv("online_shopping_session_data.csv")

# ============================================
# EXPLORATORY DATA ANALYSIS
# ============================================
print("=" * 70)
print("DATA EXPLORATION")
print("=" * 70)
print("\nFirst 5 rows:")
print(shopping_data.head())

print("\nDescriptive statistics:")
print(shopping_data.describe())

print(f"\nPurchase - Max: {shopping_data['Purchase'].max()}")
print(f"Purchase - Median: {shopping_data['Purchase'].median()}")

# ============================================
# PURCHASE RATES BY CUSTOMER TYPE (ALL DATA)
# ============================================
print("\n" + "=" * 70)
print("PURCHASE RATES BY CUSTOMER TYPE (All Data)")
print("=" * 70)
purchase_rates_all = shopping_data.groupby("CustomerType")["Purchase"].mean()
print(purchase_rates_all)

# ============================================
# CORRELATION ANALYSIS - DURATION VARIABLES (ALL DATA)
# ============================================
print("\n" + "=" * 70)
print("CORRELATION ANALYSIS - DURATION VARIABLES")
print("=" * 70)

# Single correlation
correlation_duration = shopping_data["Administrative_Duration"].corr(
    shopping_data["Informational_Duration"]
)
print(f"Administrative_Duration <-> Informational_Duration: {correlation_duration:.4f}")

# Full correlation matrix
shopping_data_corr = shopping_data.corr()
print("\nFull Correlation Matrix:")
print(shopping_data_corr)

# Max correlation (excluding diagonal)
max_corr = shopping_data_corr.mask(shopping_data_corr == 1).max().max()
print(f"\nMaximum correlation (excluding diagonal): {max_corr:.4f}")

# ============================================
# NOVEMBER & DECEMBER ANALYSIS
# ============================================
print("\n" + "=" * 70)
print("NOVEMBER & DECEMBER ANALYSIS")
print("=" * 70)

# Filter for Nov & Dec
nov_dec_data = shopping_data[shopping_data["Month"].isin(["Nov", "Dec"])]
print(f"Total sessions in Nov & Dec: {len(nov_dec_data)}")
print(f"Original dataset: {len(shopping_data)} sessions")

# ============================================
# TOTAL DURATION ANALYSIS (NOV & DEC)
# ============================================
filtered_shopping_data = nov_dec_data.copy()
filtered_shopping_data["total_duration"] = (
    filtered_shopping_data["Administrative_Duration"] + 
    filtered_shopping_data["Informational_Duration"] +
    filtered_shopping_data["ProductRelated_Duration"]
)

session_rate = filtered_shopping_data.groupby("CustomerType")["total_duration"].mean()
print("\nAverage Total Duration by Customer Type (Nov & Dec):")
print(session_rate)

# ============================================
# PURCHASE RATES BY CUSTOMER TYPE (NOV & DEC)
# ============================================
print("\n" + "=" * 70)
print("PURCHASE RATES BY CUSTOMER TYPE (Nov & Dec)")
print("=" * 70)

# Get frequency counts
session_counts = nov_dec_data.groupby(["CustomerType", "Purchase"]).size()
print("\nSession counts by CustomerType and Purchase:")
print(session_counts)

# Calculate purchase rates
customer_types = nov_dec_data["CustomerType"].unique()
purchase_rates = {}

for customer_type in customer_types:
    total_sessions = session_counts[customer_type].sum()
    purchases = session_counts.get((customer_type, 1), 0)
    purchase_rate = purchases / total_sessions if total_sessions > 0 else 0
    purchase_rates[customer_type] = purchase_rate
    
    print(f"\n{customer_type}:")
    print(f"  Total sessions: {total_sessions}")
    print(f"  Purchases: {purchases}")
    print(f"  Purchase rate: {purchase_rate:.4f} ({purchase_rate*100:.2f}%)")

# ============================================
# RETURNING CUSTOMERS CORRELATION ANALYSIS (NOV & DEC)
# ============================================
print("\n" + "=" * 70)
print("CORRELATION ANALYSIS - RETURNING CUSTOMERS (Nov & Dec)")
print("=" * 70)

# Filter for returning customers in Nov & Dec
returning_nov_dec = nov_dec_data[nov_dec_data["CustomerType"] == "Returning_Customer"]
print(f"Returning customer sessions: {len(returning_nov_dec)}")

# Duration columns
duration_cols = [
    "Administrative_Duration",
    "Informational_Duration", 
    "ProductRelated_Duration"
]

# Calculate all pairwise correlations
correlations = {}

pair_1 = ("Administrative_Duration", "Informational_Duration")
corr_1 = returning_nov_dec[pair_1[0]].corr(returning_nov_dec[pair_1[1]])
correlations[pair_1] = corr_1
print(f"\nPair 1: {pair_1[0]:30} <-> {pair_1[1]:30} = {corr_1:.4f}")

pair_2 = ("Administrative_Duration", "ProductRelated_Duration")
corr_2 = returning_nov_dec[pair_2[0]].corr(returning_nov_dec[pair_2[1]])
correlations[pair_2] = corr_2
print(f"Pair 2: {pair_2[0]:30} <-> {pair_2[1]:30} = {corr_2:.4f}")

pair_3 = ("Informational_Duration", "ProductRelated_Duration")
corr_3 = returning_nov_dec[pair_3[0]].corr(returning_nov_dec[pair_3[1]])
correlations[pair_3] = corr_3
print(f"Pair 3: {pair_3[0]:30} <-> {pair_3[1]:30} = {corr_3:.4f}")

# Find strongest correlation
strongest = max(correlations.items(), key=lambda x: abs(x[1]))

top_correlation = {
    "pair": strongest[0],
    "correlation": float(strongest[1])
}

print("\n" + "=" * 70)
print("STRONGEST CORRELATION:")
print("=" * 70)
print(f"Pair: {top_correlation['pair']}")
print(f"Correlation: {top_correlation['correlation']:.6f}")

# ============================================
# IMPROVED PURCHASE RATE CALCULATION (15% INCREASE)
# ============================================
print("\n" + "=" * 70)
print("IMPROVED PURCHASE RATE (15% Increase)")
print("=" * 70)

returning_customers_all = shopping_data[shopping_data['CustomerType'] == 'Returning_Customer']
current_purchase_rate = returning_customers_all['Purchase'].mean()
improved_purchase_rate = current_purchase_rate * 1.15

print(f"Current purchase rate (returning customers): {current_purchase_rate:.4f}")
print(f"Improved purchase rate (15% increase):       {improved_purchase_rate:.4f}")
print(f"Absolute increase:                           {improved_purchase_rate - current_purchase_rate:.4f}")

# ============================================
# BINOMIAL PROBABILITY CALCULATIONS
# ============================================
print("\n" + "=" * 70)
print("BINOMIAL PROBABILITY CALCULATIONS")
print("=" * 70)

p = shopping_data['Purchase'].mean()
n = 100

print(f"Purchase rate (p): {p:.4f}")
print(f"Number of sessions (n): {n}")

# Q1: At most 100 purchases
prob_at_most_100 = binom.cdf(k=100, n=n, p=p)
print(f"\nP(X ≤ 100) = {prob_at_most_100:.4f}")

# Q2: Exactly 15 purchases
prob_exactly_15 = binom.pmf(k=15, n=n, p=p)
print(f"P(X = 15) = {prob_exactly_15:.4f}")

# Q3: More than 20 purchases
prob_more_than_20 = 1 - binom.cdf(k=20, n=n, p=p)
print(f"P(X > 20) = {prob_more_than_20:.4f}")

# Q4: Between 10 and 20 purchases
prob_between = binom.cdf(k=20, n=n, p=p) - binom.cdf(k=9, n=n, p=p)
print(f"P(10 ≤ X ≤ 20) = {prob_between:.4f}")

# Q5: At least 100 sales (larger sample)
n_large = len(shopping_data)
prob_at_least_100_sales = 1 - binom.cdf(k=99, n=n_large, p=p)
print(f"\nP(X ≥ 100) with n={n_large}: {prob_at_least_100_sales:.6f}")

# ============================================
# BINOMIAL DISTRIBUTION VISUALIZATION
# ============================================
print("\n" + "=" * 70)
print("GENERATING BINOMIAL DISTRIBUTION PLOTS")
print("=" * 70)

# Parameters for visualization
returning_customers_all = shopping_data[shopping_data['CustomerType'] == 'Returning_Customer']
p_viz = returning_customers_all['Purchase'].mean()
n_viz = 500

# Calculate probabilities
k_values = np.arange(0, n_viz+1)
probabilities = stats.binom.pmf(k=k_values, n=n_viz, p=p_viz)

# Create subplot with zoomed view
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# LEFT PLOT: Full Distribution
ax1.bar(k_values, probabilities, color='steelblue', alpha=0.7, 
        edgecolor='black', linewidth=0.5)
ax1.axvline(x=100, color='red', linestyle='--', linewidth=2, label='k=100')
ax1.axvline(x=n_viz*p_viz, color='green', linestyle=':', linewidth=2, 
            label=f'Expected (μ={n_viz*p_viz:.1f})')
ax1.set_xlabel('Number of Sales (k)', fontsize=12)
ax1.set_ylabel('Probability', fontsize=12)
ax1.set_title(f'Full Binomial Distribution (n={n_viz}, p={p_viz:.4f})', 
              fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# RIGHT PLOT: Zoomed Around Mean
std = np.sqrt(n_viz * p_viz * (1 - p_viz))
lower = max(0, int(n_viz*p_viz - 3*std))
upper = min(n_viz, int(n_viz*p_viz + 3*std))
zoom_range = range(lower, upper+1)

ax2.bar(zoom_range, stats.binom.pmf(k=zoom_range, n=n_viz, p=p_viz), 
        color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.axvline(x=100, color='red', linestyle='--', linewidth=2, label='k=100')
ax2.axvline(x=n_viz*p_viz, color='green', linestyle=':', linewidth=2, 
            label=f'Expected (μ={n_viz*p_viz:.1f})')
ax2.set_xlabel('Number of Sales (k)', fontsize=12)
ax2.set_ylabel('Probability', fontsize=12)
ax2.set_title(f'Zoomed View (μ±3σ range)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Print distribution statistics
print(f"\nDistribution Statistics:")
print(f"Mean (μ): {n_viz*p_viz:.2f}")
print(f"Standard Deviation (σ): {std:.2f}")
print(f"Variance (σ²): {n_viz*p_viz*(1-p_viz):.2f}")
print(f"Likely range (μ±3σ): [{lower}, {upper}]")
print(f"\nP(X = 100) = {stats.binom.pmf(k=100, n=n_viz, p=p_viz):.6f}")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nKey Results:")
print(f"  Top correlation (Returning, Nov-Dec): {top_correlation}")
print(f"  Improved purchase rate: {improved_purchase_rate:.4f}")
print(f"  Probability of ≥100 sales: {prob_at_least_100_sales:.6f}")
