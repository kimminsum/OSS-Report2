# report.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Lambda function definitions
AU = lambda df: df.sum(axis=0)  # Sum of ratings
Avg = lambda df: df.mask(df==0).mean(axis=0)  # Average rating
SC = lambda df: df.gt(0).sum(axis=0)  # Count of non-zero ratings
AV = lambda df: (df >= 4).sum(axis=0)  # Count of ratings greater than or equal to 4 for each movie
BC = lambda df: ((df.shape[1] - df.rank(axis=1, method='average', ascending=False)) - 1).sum(axis=0)  # Rank-based score
CR = lambda df: np.sign(df.apply(lambda col: np.sign(df.rsub(col, axis=0)).sum(axis=0))).sum(axis=0)  # Comparative ranking

# read ratings.dat
ratings = np.genfromtxt('data/ratings.dat', delimiter='::', dtype=int)

ratings_df = pd.DataFrame(ratings, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'])

# create user-item matrix
user_item_matrix = ratings_df.pivot_table(index='UserID', columns='MovieID', values='Rating', fill_value=0)

# k-means
kmeans = KMeans(n_clusters=3, random_state=42)
user_groups = kmeans.fit_predict(user_item_matrix)

group_counts = np.bincount(user_groups)

# print group number
group_counts_df = pd.DataFrame({'Group': np.arange(1, len(group_counts) + 1), 'Number of Users': group_counts})
print(group_counts_df)


for group in range(3):
    group_users = np.where(user_groups == group)[0] + 1

    # extract data for users in the current group
    group_data = user_item_matrix[user_item_matrix.index.isin(group_users)]

    # calculate results
    res = pd.DataFrame()
    res['AU'] = AU(group_data)
    res['Avg'] = Avg(group_data)
    res['SC'] = SC(group_data)
    res['AV'] = AV(group_data)
    res['BC'] = BC(group_data)
    res['CR'] = CR(group_data)

    # extract top 10 results
    res_top10 = pd.DataFrame({col: res[col].nlargest(10).index for col in res.columns}, index=np.arange(1, 11))

    print(f"\nGroup {group + 1}:\n{res_top10}")
