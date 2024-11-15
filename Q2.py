import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from matplotlib.patches import Ellipse

# Step 1: Define the True GMM
true_gmm_components = 4
means = [
    [0, 0],
    [1.2, 1.2],  
    [5, 0],
    [0, 5]
]
covariances = [
    [[1, 0], [0, 1]],
    [[2.0, 0.8], [0.8, 2.0]],  
    [[2, 0], [0, 2]],
    [[1, -0.5], [-0.5, 1]]
]
weights = [0.4, 0.3, 0.2, 0.1]

# Function to generate data from the true GMM
def generate_gmm_data(n_samples, means, covariances, weights):
    X = []
    y = []
    for _ in range(n_samples):
        # Choose component according to weights
        component = np.random.choice(len(means), p=weights)
        # Sample from the chosen Gaussian distribution
        sample = np.random.multivariate_normal(means[component], covariances[component])
        X.append(sample)
        y.append(component)
    return np.array(X), np.array(y)

# Generate datasets with 10, 100, 1000 samples
data_sizes = [10, 100, 1000]
datasets = [generate_gmm_data(size, means, covariances, weights) for size in data_sizes]

# Visualize generated datasets with covariance ellipses
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, color='blue', **kwargs):
    """
    Plots an nstd standard deviation error ellipse based on the specified covariance matrix (cov).
    Parameters
    ----------
    cov : array-like, shape (2, 2)
        Covariance matrix.
    pos : array-like, shape (2,)
        The location of the center of the ellipse.
    nstd : float
        The radius of the ellipse in numbers of standard deviations.
    ax : matplotlib.axes.Axes, optional
        The axes upon which to plot the ellipse. Default is the current axis.
    color : str
        Color of the ellipse.
    **kwargs
        Additional keyword arguments are passed to the ellipse patch.
    """
    if ax is None:
        ax = plt.gca()

    # Eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(cov)
    # Sort by largest eigenvalue
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Compute the angle of the ellipse
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height of the ellipse
    width, height = 2 * nstd * np.sqrt(vals)

    # Draw the ellipse
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, edgecolor=color, linestyle='--', linewidth=1.5, fill=False, **kwargs)

    ax.add_patch(ellip)

# Visualize generated datasets
colors = ['blue', 'green', 'orange', 'purple']
for size, (data, labels) in zip(data_sizes, datasets):
    plt.figure()
    ax = plt.gca()
    for i in range(true_gmm_components):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], alpha=0.6, label=f'Component {i + 1}', color=colors[i])
        plot_cov_ellipse(np.array(covariances[i]), means[i], ax=ax, nstd=2, color=colors[i])
    plt.title(f"Generated Data with {size} Samples")
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()

# Step 3: Model Selection Using EM Algorithm and Cross-Validation
kfold_splits = {10: 2, 100: 10, 1000: 10}  # Adjust k-fold splits based on data size
components_range = range(1, 11)

# Function to evaluate GMM model order using cross-validation
def cross_validate_gmm(X, components_range, n_splits):
    best_n_components = None
    best_score = -np.inf
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=None)
    for n_components in components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=None)
        scores = []
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            if len(X_train) < n_components:
                continue  # Skip if the number of training samples is less than the number of components
            gmm.fit(X_train)
            score = gmm.score(X_test)
            scores.append(score)
        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_n_components = n_components
    return best_n_components

# Step 4: Repeat Experiment and Report Results
n_experiments = 200
results = {size: [] for size in data_sizes}

for _ in tqdm(range(n_experiments), desc="Repeating Experiments"):
    for size, (data, _) in zip(data_sizes, datasets):
        best_model = cross_validate_gmm(data, components_range, kfold_splits[size])
        results[size].append(best_model)

# Summarize results
summary = {}
for size in data_sizes:
    unique, counts = np.unique(results[size], return_counts=True)
    summary[size] = dict(zip(unique, counts))

# Display results
for size in data_sizes:
    print(f"Data Size: {size}")
    print("Model Order Selection Frequency:")
    for order, frequency in summary[size].items():
        print(f"  Components: {order}, Frequency: {frequency}")
    print()

# Plot results
for size in data_sizes:
    plt.figure()
    plt.pie(summary[size].values(), labels=summary[size].keys(), autopct='%1.1f%%')
    plt.title(f'Model Selection Frequency for Data Size {size}')
    plt.legend(title="Number of Components", loc='best')
    plt.show()


