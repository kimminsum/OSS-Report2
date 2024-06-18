import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# lambda function defination
AU = lambda df: df.sum(axis=0)
Avg = lambda df: df.mean(axis=0)
SC = lambda df: df.notnull().sum(axis=0)
AV = lambda df: (df >= 4).sum(axis=0)
BC = lambda df: (df.rank(axis=1) - 1).sum(axis=0)
CR = lambda df: np.sign(df.apply(lambda col: np.sign(df.rsub(col, axis=0)).sum(axis=0))).sum(axis=0)

# read ratings.dat
with open('data/ratings.dat', 'r') as file:
    lines = file.readlines()

# user x items
ratings = []
for line in lines:
    data = line.strip().split('::')
    ratings.append([int(data[0]), int(data[1]), int(data[2])])

ratings = np.array(ratings)
num_users = np.max(ratings[:, 0])
num_items = np.max(ratings[:, 1])
user_item_matrix = np.zeros((num_users, num_items))

# user x items = rating
for row in ratings:
    user_item_matrix[row[0] - 1, row[1] - 1] = row[2]

# clustering 3 groups by k-means algorithm
kmeans = KMeans(n_clusters=3, random_state=0)
user_groups = kmeans.fit_predict(user_item_matrix)

# count group users
group_counts = np.bincount(user_groups)

# print group user number information
group_counts_df = pd.DataFrame({'Group': np.arange(1, len(group_counts) + 1), 'Number of Users': group_counts})
print(group_counts_df)

# print result
for group in range(3):
    group_users = np.where(user_groups == group)[0] + 1

    # 결과 계산
    res = pd.DataFrame()
    res['AU'] = AU(pd.DataFrame(user_item_matrix))
    res['Avg'] = Avg(pd.DataFrame(user_item_matrix))
    res['SC'] = SC(pd.DataFrame(user_item_matrix))
    res['AV'] = AV(pd.DataFrame(user_item_matrix))
    res['BC'] = BC(pd.DataFrame(user_item_matrix))
    # res['CR'] = CR(pd.DataFrame(user_item_matrix))

    print(f"\nGroup {group + 1}:\n{res}")
