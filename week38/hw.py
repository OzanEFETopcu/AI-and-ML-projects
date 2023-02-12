import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Read data from file to dataframe
df = pd.read_csv('customer_data.csv', sep=";")

# Separate training data to be entered to the algorithm (X)
# Leave only species out, because we use it for result evaluation
X = df[['Sex (1=Female, 0=Male)', 'Age', 'Average monthly purchase']].copy(deep=False) # Could be used to avoid some warnings

# Scale the data for practise using Skleanr.Standardscaler
scaler = StandardScaler()

# Scale the data using the scaler
# Please note, the dataframe converts into a numpy array (not a dataframe anymore)
X = scaler.fit_transform(X)

# Empty list for SSE (Sum of Squared Errors) values
sse = []

# Try numbers from 1-10 as numbers of clusters 
for i in range(1, 21):
    # Parameters: number of clusters, "avoid random init problems"
    kmeans = KMeans(n_clusters=i, init='k-means++')
    # Train the model
    kmeans.fit(X)
    # Get and save the square distances sum (inertia_) for the model
    sse.append(kmeans.inertia_)

# # Plot a graph for SSE values to find the optimal number of clusters
# plt.plot(range(1, 21), sse)
# plt.title('The Elbow graph')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Sum of Squared Errors')
# plt.show()


# Parameters: number of clusters, "avoid random init problems" by using 'k-means++', to get same results with the teacher
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

# Train model with cluster numbers and save the custers column
y_kmeans_predict = kmeans.fit_predict(X)


# Collect the input data, the prdicted clusters, and the known species into same dataframe
# Add original data to the dataframe
test_results = df

# Add Predicted species to the dataframe
test_results['no_clue'] = y_kmeans_predict



# # Print crosstab to see how the predicted clusters match with the known clusters (species)
# cross_tab = pd.crosstab(test_results['predicted_spendings'], test_results['Average monthly purchase'])
# print('Crosstab:\n', cross_tab)


# # Print also original species counts and predicted counts for comparison
# print('\nSpendings:\n', test_results['Average monthly purchase'].value_counts())
# print('\nPredicted Spendings:\n', test_results['predicted_spendings'].value_counts())


# Inverse scale the scaled values back to original values
X = scaler.inverse_transform(X)


# Visualise (plot) the clusters just to see the clusters
plt.scatter(X[y_kmeans_predict == 0,1], X[y_kmeans_predict == 0,2], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans_predict == 1,1], X[y_kmeans_predict == 1,2], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans_predict == 2,1], X[y_kmeans_predict == 2,2], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans_predict == 3,1], X[y_kmeans_predict == 3,2], s=100, c='yellow', label='Cluster 4')
plt.scatter(X[y_kmeans_predict == 4,1], X[y_kmeans_predict == 4,2], s=100, c='purple', label='Cluster 5')



# Finalise the graph with information
plt.title('Clusters of customers')
plt.xlabel('Age')
plt.ylabel('Spending')
plt.legend()
plt.show()
