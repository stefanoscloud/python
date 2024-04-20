# Yellowbrick Python library

# Generate an elbow analysis and visualize silhouette scores in clustering models
# visualizer = yellowbrick.cluster.KElbowVisualizer(model, k = (1, 10)) —This generates an elbow point visualization object by passing in a scikit-learn model and a range of k values to show.
# visualizer = yellowbrick.cluster.SilhouetteVisualizer(model) —This generates a silhouette visualization object by passing in a scikit-learn model. It will show the silhouette scores for the number of clusters specified in the model object that you pass in as model.
# visualizer.fit(X_train) —Fit a set of training data to the visualizer.
