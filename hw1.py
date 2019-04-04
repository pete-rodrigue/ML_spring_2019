'''
Machine learning HW 1
Spring 2019
pete rodrigue
'''

import pandas as pd
from sodapy import Socrata

# Uncomment the code below to download the crimes data again.
# Otherwise skip down to the part where we load the crimes data from the repo.

'''
# Much of the code below comes from the documentation on the
# city of chicago api docs webpage:
# https://dev.socrata.com/foundry/data.cityofchicago.org/6zsd-86xi

# Unauthenticated client only works with public data sets. Note 'None'
# in place of application token, and no username or password:
client = Socrata("data.cityofchicago.org", None)

# First 2000 results, returned as JSON from API / converted to Python list of
# dictionaries by sodapy.
results = client.get("6zsd-86xi"  # , limit=10
                     )

# Convert to pandas DataFrame
results_df = pd.DataFrame.from_records(results)
results_df.to_csv('alleged_crimes_data.csv', index=False)
'''

df = pd.read_csv('alleged_crimes_data.csv')
print(df.head(15))
print(df.shape)
print(df.columns)
