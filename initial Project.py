import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#task 1:
# Load the dataset
# I just uploaded the csv file to the GitHub repository and read it from there, so you too can access it without any issues.
url_listing_berlin = "https://raw.githubusercontent.com/Nill2nn/Statistical-Programming/refs/heads/Nill2nn-patch-excel/listings_berlin.csv"
listings_berlin = pd.read_csv(url_listing_berlin)

print("listing Berlin shape:", listings_berlin.shape)

# the first few rows of the dataset
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# I used the above two lines to make sure that all columns are displayed. pandas is truncating the display.
print(listings_berlin.head(10))

# Check for missing values
missing_number = listings_berlin.isnull().sum()
missing_percent = (missing_number/ len(listings_berlin)) * 100

missing_df = pd.DataFrame({
    'Missing in number ': missing_number,
    'Missing in Percentage': missing_percent.round(2)
})

print(missing_df)

#task 2:
"""
 the three value with the highest percentage of missing values are: 'price', 'host_response_rate' and 'review_scores' for rating and cleanliness.
I will keep and impute price column, because pricing information is crucial for our analysis and dropping it would lead to a significant loss of valuable insights. Instead, I will impute the missing values in the price column using the median price of the listings, which is a common approach for handling missing numerical data.
I will drop the 'host_response_rate' column because it has a high percentage of missing values and may not be essential for our analysis. The 'host_response_rate' won't actually play a significant role in decision-making of customers.
I will keep the missing values in the 'review_scores' for rating and cleanliness columns, as they may still provide valuable insights for customers . we could not impute these values because they are random and imputing them with mean or median could lead to misleading information.
"""

#Task 3:
#cleaning the price column by removing the $ sign and converting it to numeric
def clean_price(price_str):
    if pd.isna(price_str):
        return None
    if isinstance(price_str, str):
        return float(price_str.replace('$', '').replace(',', ''))
    return float(price_str)

listings_berlin['price'] = listings_berlin['price'].apply(clean_price)
print(listings_berlin['price'].describe())  # with .describe methode we can check minimum, median(50%), mean, and maximum

#task 4:
#Turning the bathrooms_text column into a numeric column called bathrooms
listings_berlin['bathrooms_text'].value_counts()


def clean_baths(bathroom):
    if pd.isna(bathroom):
        return None

    bathroom = str(bathroom).lower()

    if 'half' in bathroom:
        return 0.5
    else:
        numpart = bathroom.split(' ')[0]
        return float(numpart)


listings_berlin['bathrooms'] = listings_berlin['bathrooms_text'].apply(clean_baths)
print('the Types of rooms available:', listings_berlin['bathrooms'].unique())

#Task 5:
#Truning T/F into boolean values
listings_berlin['host_is_superhost'] = listings_berlin['host_is_superhost'] == 't'
listings_berlin['instant_bookable'] = listings_berlin['instant_bookable'] == 't'

print(listings_berlin['host_is_superhost'].dtype)
print(listings_berlin['instant_bookable'].dtype)
print(listings_berlin['host_is_superhost'].unique())

#The number of years the host join Airbnb
lb = listings_berlin.copy()
lb['host_tenure_years'] = 2025 - pd.to_datetime(lb['host_since']).dt.year
print(lb['host_tenure_years'].head(5))

#Task 6:
#comparing the distribution of raw price and log-transformed price

plt.figure(figsize=(14, 6))

# Raw price distribution
plt.subplot(1, 2, 1)
plt.hist(listings_berlin['price'], bins=50, color='purple', edgecolor='black')
plt.title('Raw Price Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Price', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.75)

# Log-transformed price distribution
plt.subplot(1, 2, 2)
plt.hist(np.log(listings_berlin['price']), bins=50, color='purple', edgecolor='black')
plt.title('Log-Transformed Price Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Log-Transformed Price', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.75)

plt.tight_layout()
plt.show()

""" explanation: The raw price is skewed due to outliers, making it hard to analyze and showing undesirable statistical properties.
Log transformation fixes this by creating a bell curve, which is much more suitable for statistical analysis.
With a normal distribution, our data is ready to be analyzed."""

#task 7:
# sorting the neighborhoods_cleansed column by  median price in descending order
neighborhood_summary = (listings_berlin.groupby('neighbourhood_cleansed').agg(
    count=('neighbourhood_cleansed', 'count'),
    median_price=('price', 'median'),
    mean_rating=('review_scores_rating', 'mean')
).round(2).sort_values('median_price', ascending=False))

print(neighborhood_summary.head(3))
""" The top three neighborhoods with the highest median price are:
 1. Haselhorst with a median price of 236.5 and average rating of 4.86
 2. West 5 with a median price of 203 and average rating of 4.79
 3. Schmargendorf  with a median price of 172 and average rating of 4.66 """


#task 9:
#drawing plots to show the relationship between price and room_type, price and host status and price and review_score_rating
plt.figure(figsize=(12, 6))
# Price vs room_type
plt.subplot(1, 2, 1)
# gathering and collecting the prices of each type of room together to plot them in the same graph
for room in listings_berlin['room_type'].unique():
    subset = listings_berlin[listings_berlin['room_type'] == room]
    sns.kdeplot(subset['price'], label=room, fill=False, alpha=0.6)

plt.title('Price Distribution by Room Type', fontsize=14, fontweight='bold')
plt.xlabel('Price (€)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlim(0, 500)
plt.legend(title='Room Type')
plt.grid(axis='y', alpha=0.3)

#  to compare the price distribution between superhosts and regular hosts
plt.subplot(1, 2, 2)

for superhost in listings_berlin['host_is_superhost'].unique():
    subset = listings_berlin[listings_berlin['host_is_superhost'] == superhost]
    label = 'Superhost' if superhost == True else 'Regular Host'
    sns.kdeplot(subset['price'], label=label, fill=False, alpha=0.4)

plt.title('Price Distribution by Superhost Status', fontsize=14, fontweight='bold')
plt.xlabel('Price (€)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlim(0, 500)
plt.legend(title='Host Type')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Price vs review_score_rating
plt.figure(figsize=(10, 6))
# Remove missing values
data_clean = listings_berlin.dropna(subset=['review_scores_rating', 'price'])
# Remove extreme price outliers
data_clean = data_clean[data_clean['price'] < 500]
# Remove zeros in review scores
data_clean = data_clean[data_clean['review_scores_rating'] > 0]
sns.regplot(data=data_clean, x='review_scores_rating', y='price',
            scatter_kws={'alpha': 0.3, 'color': 'steelblue'},  # change to steelblue
            line_kws={'color': 'red', 'linewidth': 2})
plt.xlabel('Review Score Rating', fontsize=12)
plt.ylabel('Price (€)', fontsize=12)
plt.ylim(0, 300)
plt.title('Price vs Review Score Rating', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.show()

#task 10:
""" 1. The price distribution for different room types shows that private rooms have the most density  at a price-range of around 50-100 euros,
while entire homes/apartments have a wider price distribution with a peak around 100-150 euros. Shared rooms are less common and have a lower price range, mostly below 50 euros.
    2. the price distribution for superhosts shows that superhosts have the most density at a price range of around 100-150 euros, while regular hosts have a wider price distribution.
This suggests that superhosts tend to charge higher prices compared to regular hosts.
    3. Listings with higher review scores tend to charge slightly higher prices, but the relationship is very weak.
    Most listings are highly rated between 4 and 5. 3."""