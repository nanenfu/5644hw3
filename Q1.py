import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Set random seed to ensure reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(12345)  # Set random seed

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Data Distribution and Generation
means = [
    np.array([2.5, 0, 0]),
    np.array([0, 2.5, 0]),
    np.array([0, 0, 2.5]),
    np.array([-2.5, -2.5, -2.5])
]
covs = [
    np.array([[2.5, 0, 0], [0, 2.5, 0], [0, 0, 2.5]]) for _ in range(4)
]

# Data generation function
def generate_data(mean, cov, num_samples, label):
    data = multivariate_normal.rvs(mean=mean, cov=cov, size=num_samples)
    labels = np.full(num_samples, label)
    return data, labels

# Create training and testing datasets
def create_datasets(num_samples_list, test_samples):
    train_data = {}
    for n_samples in num_samples_list:
        data, labels = [], []
        for i, (mean, cov) in enumerate(zip(means, covs)):
            d, l = generate_data(mean, cov, n_samples // 4, i)
            data.append(d)
            labels.append(l)
        train_data[n_samples] = (np.vstack(data), np.hstack(labels))

    data, labels = [], []
    for i, (mean, cov) in enumerate(zip(means, covs)):
        d, l = generate_data(mean, cov, test_samples // 4, i)
        data.append(d)
        labels.append(l)
    test_data = (np.vstack(data), np.hstack(labels))
    return train_data, test_data

num_samples_list = [100, 500, 1000, 5000, 10000]
test_samples = 100000
train_data, test_data = create_datasets(num_samples_list, test_samples)

# Step 2: MLP Model Structure
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.activation = nn.ELU()
        self.output = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.softmax(self.output(x))
        return x

# Step 4: Theoretically Optimal Classifier
def theoretical_optimal_classifier(data, means, covs, priors):
    likelihoods = []
    for mean, cov in zip(means, covs):
        likelihood = multivariate_normal.pdf(data, mean=mean, cov=cov)
        likelihoods.append(likelihood)
    likelihoods = np.array(likelihoods).T * priors
    return np.argmax(likelihoods, axis=1)

priors = [0.25] * 4
test_data_values, test_labels = test_data
pred_labels = theoretical_optimal_classifier(test_data_values, means, covs, priors)
theoretical_error = np.mean(pred_labels != test_labels)
print(f"Theoretical Optimal Classifier Error Rate: {theoretical_error}")

# Step 5: Model Order Selection using Cross-Validation
def cross_validate(train_data, hidden_sizes, num_folds=10, epochs=300):
    results = {}
    for n_samples, (data, labels) in train_data.items():
        kf = KFold(n_splits=num_folds)
        best_hidden_size = None
        best_accuracy = 0
        
        for hidden_size in hidden_sizes:
            fold_accuracies = []
            for train_idx, val_idx in kf.split(data):
                x_train, y_train = data[train_idx], labels[train_idx]
                x_val, y_val = data[val_idx], labels[val_idx]

                model = MLP(input_size=3, hidden_size=hidden_size, num_classes=4).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters())

                x_train, y_train = torch.Tensor(x_train).to(device), torch.LongTensor(y_train).to(device)
                x_val, y_val = torch.Tensor(x_val).to(device), torch.LongTensor(y_val).to(device)

                # Train model with tqdm progress bar
                model.train()
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    outputs = model(x_train)
                    loss = criterion(outputs, y_train)
                    loss.backward()
                    optimizer.step()

                # Validate model
                model.eval()
                with torch.no_grad():
                    outputs = model(x_val)
                    _, predicted = torch.max(outputs, 1)
                    accuracy = (predicted == y_val).sum().item() / y_val.size(0)
                    fold_accuracies.append(accuracy)
            
            avg_accuracy = np.mean(fold_accuracies)
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_hidden_size = hidden_size
        
        results[n_samples] = best_hidden_size
    return results

hidden_sizes = [30, 40, 60]
best_hidden_sizes = cross_validate(train_data, hidden_sizes)
print("Best hidden sizes for each dataset size:", best_hidden_sizes)

# Step 6: Model Training with Selected Hidden Size
def train_model(data, labels, hidden_size):
    model = MLP(input_size=3, hidden_size=hidden_size, num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    x_train, y_train = torch.Tensor(data).to(device), torch.LongTensor(labels).to(device)
    
    model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    return model

trained_models = {}
for n_samples, (data, labels) in train_data.items():
    hidden_size = best_hidden_sizes[n_samples]
    model = train_model(data, labels, hidden_size)
    trained_models[n_samples] = model

# Step 7: Performance Assessment on Test Set
def evaluate_model(model, data, labels):
    x_test, y_test = torch.Tensor(data).to(device), torch.LongTensor(labels).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(x_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        error_rate = 1 - accuracy
    return error_rate

error_rates = {n_samples: evaluate_model(model, test_data_values, test_labels) for n_samples, model in trained_models.items()}
print("Error rates for each dataset size on test set:", error_rates)

# Step 8: Report Process and Results (Plotting)
sizes = list(error_rates.keys())
errors = list(error_rates.values())

plt.plot(sizes, errors, marker='o', linestyle='-')
plt.xscale('log')
plt.xlabel("Training Set Size")
plt.ylabel("Empirical Error Rate")
plt.title("MLP Model Error Rates vs Training Set Size")
plt.axhline(y=theoretical_error, color='r', linestyle='--', label="Theoretical Optimal Classifier Error")
plt.legend()
plt.show()
