from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.clustering import KMeans
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from pyspark.mllib.linalg import Vectors
import pandas as pd
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from scipy.linalg import svd
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.feature import StandardScaler
import plotly.express as px
from pyspark.mllib.stat import MultivariateStatisticalSummary
import math
from math import pi
from math import sqrt


rdd1 = sc.textFile("space.dat")
rdd2 = rdd1.map(lambda x: x.split(','))
rdd3 = rdd2.map(lambda x: [float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])])

# Train KMeans model and compute cost for different cluster numbers
for clusters in range(1, 15):
    model = KMeans.train(rdd3, clusters, initializationMode="k-means||", seed=888, epsilon=1e-4)
    print(clusters, model.computeCost(rdd3))
#k=6 The decrease is obvious

# Train KMeans model with 6 clusters
model = KMeans.train(rdd3, 6, initializationMode="k-means||", seed=888, epsilon=1e-4)
labels = rdd3.map(lambda x: model.predict(x)).collect()
rdd_with_labels = rdd3.map(lambda x: (x, model.predict(x)))
rdd_with_labels_transformed = rdd_with_labels.map(lambda x: (x[0] + [x[1]]))
#the plot of all clusters
import plotly.express as px
import pandas as pd
data_points = rdd_with_labels_transformed.collect()
df = pd.DataFrame(data_points, columns=['axis1', 'axis2', 'axis3','axis4', 'axis5', 'axis6','color'])
# Create a 3D scatter plot
fig = px.scatter_3d(df, x='axis1', y='axis2', z='axis3',color='color')
fig.update_traces(marker=dict(size=3))
fig.update_layout(title="3D PCA Plot for Cluster 4 (PC1, PC2, PC3)")
fig.write_html("/jet/home/ywu22/cluster_data_overall_3D_all.html")

#delete outliers
rdd_filtered = rdd3.filter(lambda x: x[0] <= 90 and x[1] <= 90 and x[2] <= 90)
rdd_with_labels = rdd_filtered.map(lambda x: (x, model.predict(x)))
rdd_with_labels_transformed = rdd_with_labels.map(lambda x: (x[0] + [x[1]]))

# Loop over cluster labels (0 to 5)
for cluster_id in range(6):
    # Filter RDD by cluster label
    filtered_rdd = rdd_with_labels_transformed.filter(lambda x: x[-1] == cluster_id)
    filtered_rdd_cluster = filtered_rdd.map(lambda x: x[:-1])
    #center the data
    sums = filtered_rdd_cluster.reduce(lambda x, y: [x[i] + y[i] for i in range(len(x))])
    n = filtered_rdd_cluster.count()
    print(f"count of data in cluster{cluster_id}:", n)
    means = [sum_val / n for sum_val in sums]
    centered_rdd = filtered_rdd_cluster.map(lambda x: [x[i] - means[i] for i in range(len(x))])
    centeredmat = RowMatrix(centered_rdd) 
    #pca for each cluster
    mat = RowMatrix(filtered_rdd_cluster)
    # Compute principal components
    pc = mat.computePrincipalComponents(6)
    # Save or print the principal components for this cluster
    print(f"PC for cluster {cluster_id}:", pc)
    # Project data onto the principal components
    projected = mat.multiply(pc)

    # Perform SVD
    svd = centeredmat.computeSVD(6, computeU=True)
    U = svd.U
    s = svd.s
    V = svd.V
    evs=s*s
    variances = evs/sum(evs)
    print(f"variances:{variances}")# determine the dimension of the object

    # Project data onto the first two principal components for visualization
    projected_2d = mat.multiply(pc).rows.map(lambda x: (x[0], x[1])).collect()
    
    # 2D Plot
    x_vals, y_vals = zip(*projected_2d)
    plt.scatter(x_vals, y_vals)
    plt.title(f"2D Projection of Cluster {cluster_id} in 6D Space")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig(f"kmeansresult_cluster{cluster_id}_2d.png")
    plt.clf()  # Clear the plot to avoid overlap in subsequent iterations
    
    # 3D Plot
    projected_3d = mat.multiply(pc).rows.map(lambda x: (x[0], x[1], x[2])).collect()
    fig = px.scatter_3d(projected_3d, x=0, y=1, z=2)
    
    df = pd.DataFrame(projected_3d, columns=['PC1', 'PC2', 'PC3'])
    fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3')
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(title=f"3D PCA Plot for Cluster {cluster_id} (PC1, PC2, PC3)")
    fig.write_html(f"/jet/home/ywu22/cluster_{cluster_id}_3D_PCA.html")
    
    # Compute center of the original points (in the original 6D space)
    summary = mat.computeColumnSummaryStatistics()
    center_position = summary.mean()
    print(f"Center of cluster {cluster_id} in original 6D space:", center_position)
    
    # Save the center position
    with open(f"center_position_cluster_{cluster_id}.txt", "w") as f:
        f.write(str(variances))
        f.write(str(center_position))
    if cluster_id == 0:#3d cube
        #length
        first_component = [row[0] for row in projected_3d]
        second_component = [row[1] for row in projected_3d]
        third_component = [row[2] for row in projected_3d]
        length1 = max(first_component) - min(first_component)
        length2 = max(second_component) - min(second_component)
        length3 = max(third_component) - min(third_component)
        print(f"Length of first component: {length1}, Length of second component: {length2}, Length of third component: {length3}")

        # orientation
        directions = np.array(pc.toArray())[:, :3]
        angles = np.arccos(directions / np.linalg.norm(directions, axis=0))  
        angles = np.degrees(angles)  
        print(angles)
    if cluster_id ==1:#sphere
        projected_3d = mat.multiply(pc).rows.map(lambda x: (x[3], x[4],x[5])).collect()
        df = pd.DataFrame(projected_3d, columns=['PC3', 'PC4', 'PC5'])
        # Create a 3D scatter plot double check
        fig = px.scatter_3d(df, x='PC3', y='PC4', z='PC5')
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(title="3D PCA Plot for Cluster 2 (PC3, PC4, PC5)")
        fig.write_html("/jet/home/ywu22/cluster_data_2_2_3D.html")
        # Compute the variance of each dimension to check for symmetry
        summary = mat.computeColumnSummaryStatistics()
        variances = summary.variance()
        print("Variances across dimensions:", variances)#remain steady in the range 6.6 to6.9

        #ditribution of distance to center
        center = summary.mean()  
        points = np.array(mat.rows.collect())
        distances = np.linalg.norm(points - center, axis=1)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        print(f"Mean Distance to Center: {mean_distance}")
        print(f"Standard Deviation of Distances: {std_distance}")

        plt.hist(distances, bins=30, alpha=0.7, color='blue')
        plt.xlabel("Distance to Center")
        plt.ylabel("Frequency")
        plt.title("Distribution of Distances to Center")
        plt.show()
        plt.savefig(f"distribution of cluster2.png")#according to the distribution and the varience we conclude that it is a Hypersphere
        # 3.radius
        positive_distances = [d for d in distances if d > 0]
        estimated_radius = np.mean(positive_distances)
        print("Estimated radius of the hypersphere:", estimated_radius)
    if cluster_id ==2:#line
        #length 
        first_component = [row[0] for row in projected_2d]
        length = max(first_component) - min(first_component)
        print(f"Length of the line in original space: {length}")
        # directon
        pc_array = pc.toArray()
        direction = pc_array[:, 0]
        direction_norm = direction / np.linalg.norm(direction)  
        print("Direction of the line in original space", direction_norm)

    if cluster_id ==3:#6D branches (more like a star in 6D)
        #2. average length of branches
        print("center location:", center_position)
        points = np.array(mat.rows.collect())
        distances = np.linalg.norm(points - center_position, axis=1)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        print(f"Mean Distance to Center: {mean_distance}")
        print(f"Standard Deviation of Distances: {std_distance}")

    if cluster_id ==4:
        # Ellipsoid size
        directions = np.array(pc.toArray())[:, :3]
        volume = (4 / 3) * pi * sqrt(evs[0]) * sqrt(evs[1]) * sqrt(evs[2])
        print("Ellipsoid size:", volume)
        # orientation
        angles = np.arccos(directions / np.linalg.norm(directions, axis=0)) 
        angles = np.degrees(angles)  
        print(angles)

    if cluster_id ==5:#6 Quasi-cuboid (parallelogram)
        #length
        first_component = [row[0] for row in projected_3d]
        second_component = [row[1] for row in projected_3d]
        third_component = [row[2] for row in projected_3d]
        length1 = max(first_component) - min(first_component)
        length2 = max(second_component) - min(second_component)
        length3 = max(third_component) - min(third_component)
        print(f"Length of first component: {length1}, Length of second component: {length2}, Length of third component: {length3}")

        # orientation
        directions = np.array(pc.toArray())[:, :3]
        angles = np.arccos(directions / np.linalg.norm(directions, axis=0))  
        angles = np.degrees(angles)  
        print(angles)
