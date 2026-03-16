# Predicting Hotel Cancellations

## Background

A hotel wants to increase revenue from room bookings by using data science
to reduce cancellations. The goal is to identify the factors that contribute
to whether a booking will be fulfilled or cancelled, and to produce
actionable recommendations.
## Data Dictionary

| Column | Description |
|--------|-------------|
| `Booking_ID` | Unique identifier of the booking |
| `no_of_adults` | Number of adults |
| `no_of_children` | Number of children |
| `no_of_weekend_nights` | Weekend nights (Sat/Sun) |
| `no_of_week_nights` | Week nights (Mon–Fri) |
| `type_of_meal_plan` | Meal plan included |
| `required_car_parking_space` | Car parking space required (0/1) |
| `room_type_reserved` | Room type reserved |
| `lead_time` | Days between booking and arrival |
| `arrival_year` | Year of arrival |
| `arrival_month` | Month of arrival |
| `arrival_date` | Day of the month |
| `market_segment_type` | How the booking was made |
| `repeated_guest` | Whether the guest previously stayed (0/1) |
| `no_of_previous_cancellations` | Count of previous cancellations |
| `no_of_previous_bookings_not_canceled` | Count of previous non-cancelled bookings |
| `avg_price_per_room` | Average price per night |
| `no_of_special_requests` | Count of special requests |
| `booking_status` | `Canceled` or `Not_Canceled` |

**Source:** [Kaggle – Hotel Reservations Classification Dataset](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset)
## 1. Setup & Data Loading
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Plot defaults
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 7)
hotels = pd.read_csv("data/hotel_bookings.csv")
print(f"Shape: {hotels.shape}")
hotels.head()
hotels.info()
hotels.describe()
## 2. Feature Engineering
# Binary target: 1 = Not Canceled, 0 = Canceled
hotels["is_not_canceled"] = (
    hotels["booking_status"]
    .map({"Not_Canceled": 1, "Canceled": 0})
)

# Family size
hotels["family_size"] = hotels["no_of_adults"] + hotels["no_of_children"]

hotels[["booking_status", "is_not_canceled", "family_size"]].head()
## 3. Exploratory Data Analysis
# Booking status distribution
print(hotels["booking_status"].value_counts())
print()
print(hotels["booking_status"].value_counts(normalize=True).round(3))
### 3.1 Correlation with Booking Status
# Correlation of numeric features with the binary target
numeric_cols = hotels.select_dtypes(include="number").columns
target_corr = (
    hotels[numeric_cols]
    .corr()["is_not_canceled"]
    .drop("is_not_canceled")
)

top_correlations = (
    target_corr
    .abs()
    .sort_values(ascending=False)
    .head(10)
)

print("Top 10 correlations with booking status (absolute):")
print("=" * 60)
for col in top_correlations.index:
    print(f"  {col:40s} {target_corr[col]:+.4f}")
### 3.2 Top Inter-Feature Correlations (Cancelled Bookings Only)
canceled = hotels[hotels["booking_status"] == "Canceled"]
canceled_corr = canceled.select_dtypes(include="number").corr()

# Upper triangle to avoid duplicates
mask = np.triu(np.ones_like(canceled_corr, dtype=bool), k=1)
top_pairs = (
    canceled_corr
    .where(mask)
    .stack()
    .sort_values(ascending=False)
    .head(10)
)

print("Top 10 feature-pair correlations (cancelled bookings):")
print("=" * 60)
for (col1, col2), value in top_pairs.items():
    print(f"  {col1:25s} <-> {col2:25s}  {value:+.4f}")
### 3.3 Lead Time vs. Booking Volume
lead_time_counts = (
    hotels
    .groupby(["lead_time", "booking_status"])
    .size()
    .reset_index(name="total_bookings")
)

plt.figure()
sns.scatterplot(
    data=lead_time_counts,
    x="lead_time",
    y="total_bookings",
    hue="booking_status",
    palette="Set2",
    s=120,
    alpha=0.7,
    edgecolor="black",
    linewidth=0.5,
)
plt.title("Total Bookings by Lead Time and Status", fontsize=16, fontweight="bold")
plt.xlabel("Lead Time (days)")
plt.ylabel("Total Bookings")
plt.legend(title="Status")
plt.tight_layout()
plt.show()
### 3.4 Family Size vs. Cancellation
family_corr = hotels["family_size"].corr(hotels["is_not_canceled"])
print(f"Correlation (family_size <-> is_not_canceled): {family_corr:.4f}")

family_counts = (
    hotels
    .groupby(["family_size", "booking_status"])
    .size()
    .reset_index(name="total_bookings")
)

plt.figure()
sns.scatterplot(
    data=family_counts,
    x="family_size",
    y="total_bookings",
    hue="booking_status",
    palette="Set2",
    s=150,
    alpha=0.7,
    edgecolor="black",
    linewidth=0.5,
)
plt.title("Total Bookings by Family Size and Status", fontsize=16, fontweight="bold")
plt.xlabel("Family Size")
plt.ylabel("Total Bookings")
plt.legend(title="Status")
plt.tight_layout()
plt.show()
### 3.5 Market Segment vs. Cancellation
segment_counts = (
    hotels
    .groupby(["market_segment_type", "booking_status"])
    .size()
    .reset_index(name="total_bookings")
)

plt.figure()
sns.barplot(
    data=segment_counts,
    x="market_segment_type",
    y="total_bookings",
    hue="booking_status",
    palette="Set2",
    edgecolor="black",
)
plt.title("Bookings by Market Segment and Status", fontsize=16, fontweight="bold")
plt.xlabel("Market Segment")
plt.ylabel("Total Bookings")
plt.legend(title="Status")
plt.tight_layout()
plt.show()
