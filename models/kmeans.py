import torch
import torch.nn as nn

class Kmeans(nn.Module):
    def __init__(self, k, n_iters=10):
        super(Kmeans, self).__init__()
        self.k = k 
        self.n_iters = n_iters
        self.centroids = None
        
    def fit(self, X):
        # Initialize centroids randomly
        centroids = X[torch.randperm(X.size(0))[:self.k]]
    
        for _ in range(self.n_iters):
            # Assign each data point to the nearest centroid
            distances = torch.cdist(X, centroids)
            assignments = torch.argmin(distances, dim=1)
            
            # Update centroids
            new_centroids = torch.stack([X[assignments == i].mean(0) for i in range(self.k)])
        
            # Check for convergence
            if torch.all(torch.abs(centroids - new_centroids) < 1e-5):
                break
            centroids = new_centroids
        self.centroids = centroids
    
    def forward(self,X):
        dist = torch.cdist(X, self.centroids)
        return torch.argmin(dist, dim=1)



if __name__ == "__main__":
    ## Check if it works
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    
    # Set random seed for reproducibility
    torch.manual_seed(39)
    
    # Generate a 2D dataset with 3 clusters for training
    X_train, _ = make_blobs(n_samples=300, centers=3, random_state=42)
    
    # Convert the NumPy array to a PyTorch tensor
    X_train_tensor = torch.FloatTensor(X_train)
    
    # Create an instance of the Kmeans class
    kmeans_model = Kmeans(k=3, n_iters=100)
    
    # Fit the model to the training data
    kmeans_model.fit(X_train_tensor)
    
    # Generate a regularly spaced grid for testing
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 100), torch.linspace(y_min, y_max, 100))
    
    # Flatten the grid and concatenate the coordinates
    grid_tensor = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    # Get cluster assignments for the grid using the forward method
    grid_labels = kmeans_model(grid_tensor)
    
    # Reshape the labels to match the shape of the meshgrid
    grid_labels = grid_labels.view(xx.shape)
    
    # Plot the training data and centroids
    plt.scatter(X_train[:, 0], X_train[:, 1], c=kmeans_model(X_train_tensor).numpy(), cmap='viridis')
    plt.scatter(kmeans_model.centroids[:, 0], kmeans_model.centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    
    # Plot the decision boundaries on the grid
    plt.contourf(xx.numpy(), yy.numpy(), grid_labels.numpy(), alpha=0.3, cmap='viridis')
    
    plt.title('K-Means Clustering Decision Boundaries')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


        