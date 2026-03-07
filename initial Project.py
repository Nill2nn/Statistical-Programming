import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
# the three value with the highest percentage of missing values are: 'price', 'host_response_rate' and 'review_scores' for rating and cleanliness.
# I will keep and impute price column, because pricing information is crucial for our analysis and dropping it would lead to a significant loss of valuable insights. Instead, I will impute the missing values in the price column using the median price of the listings, which is a common approach for handling missing numerical data.
# I will drop the 'host_response_rate' column because it has a high percentage of missing values and may not be essential for our analysis. The 'host_response_rate' won't actually play a significant role in decision-making of customers.
# I will keep the missing values in the 'review_scores' for rating and cleanliness columns, as they may still provide valuable insights for customers about the situation of the Airbnbs. we could not impute these values because they are ratings and imputing them with mean or median could lead to misleading information.