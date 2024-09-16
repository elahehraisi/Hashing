import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import hashlib


class HybridHashing:
    def __init__(self, top_k, num_buckets, embedding_dim):
        # Number of most frequent IDs to hash uniquely
        self.top_k = top_k
        # Size of the hash space for low-frequency IDs
        self.num_buckets = num_buckets
        # Dimensionality of the embeddings
        self.embedding_dim = embedding_dim

        # Dictionary to map the top-K frequent IDs to their unique indices
        self.freq_dict = {}

        # Embedding parameters for top-K frequent features (trainable)
        self.top_k_embeddings = nn.Embedding(top_k, embedding_dim)

        # Embedding parameters for double hashing of low-frequency features (trainable)
        self.low_freq_embeddings = nn.Embedding(num_buckets, embedding_dim)

    def find_top_k_frequent(self, data):
        """Find the top-K most frequent features (IDs) in the data."""
        frequency = defaultdict(int)
        for user_id in data:
            frequency[user_id] += 1

        # Sort user_ids by frequency, and take the top K most frequent ones
        sorted_ids = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        self.freq_dict = {user_id: idx for idx, (user_id, _) in enumerate(sorted_ids[:self.top_k])}

    def hash_function_1(self, f):
        """Applies the first hash function (h1) using MD5."""
        return int(hashlib.md5(f.encode()).hexdigest(), 16) % self.num_buckets

    def hash_function_2(self, f):
        """Applies the second hash function (h2) using SHA1."""
        return int(hashlib.sha1(f.encode()).hexdigest(), 16) % self.num_buckets

    def get_embedding(self, user_id):
        """Hybrid hashing to retrieve embedding."""
        if user_id in self.freq_dict:
            # Use single hashing (unique embedding) for top-K frequent IDs
            unique_index = self.freq_dict[user_id]
            return self.top_k_embeddings(torch.tensor(unique_index))
        else:
            # Use double hashing for less frequent IDs
            h1_code = self.hash_function_1(user_id)
            h2_code = self.hash_function_2(user_id)
            E_h1 = self.low_freq_embeddings(torch.tensor(h1_code))
            E_h2 = self.low_freq_embeddings(torch.tensor(h2_code))

            # Aggregate the embeddings (e.g., element-wise summation)
            return E_h1 + E_h2


class SimpleModel(nn.Module):
    def __init__(self, embedding_dim):
        super(SimpleModel, self).__init__()
        # Simple linear layer for prediction, predict a single value
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        return self.fc(x)


# Example Training Data (user_id's and some labels)
user_data = [1234, 5678, 1234, 9876, 1111, 1234, 2222, 5678, 3333, 2222, 4444]
labels = torch.tensor([5, 3, 5, 2, 4, 5, 3, 3, 4, 3, 2], dtype=torch.float32)  # Example labels

# Initialize HybridHashing with parameters
# Top 100 most frequent IDs get unique embeddings
top_k = 100

# Hash space for low-frequency IDs
num_buckets = 100

# Dimension of the embedding vector
embedding_dim = 32

# Instantiate the hybrid hashing model
hybrid_hashing = HybridHashing(top_k, num_buckets, embedding_dim)

# Find the top-K frequent user IDs in the data
hybrid_hashing.find_top_k_frequent(user_data)

# Define a simple model that will use the embeddings
model = SimpleModel(embedding_dim)

# Define optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training Loop
epochs = 100
for epoch in range(epochs):
    total_loss = 0
    for i, user_id in enumerate(user_data):
        # Get the hybrid embedding for the user_id
        embedding = hybrid_hashing.get_embedding(user_id)

        # Forward pass: Compute prediction
        prediction = model(embedding)

        # Compute loss
        loss = criterion(prediction, labels[i].unsqueeze(0))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}')

# Testing the embedding for a new user_id
test_user_id = 5678
test_embedding = hybrid_hashing.get_embedding(test_user_id)
print(f"Embedding for test_user_id {test_user_id}: {test_embedding}")
