import torch
import torch.nn as nn
import torch.optim as optim
import hashlib


def hash_function(value, num_buckets):
    hash_value = int(hashlib.md5(str(value).encode('utf-8')).hexdigest(), 16)
    return hash_value % num_buckets

class SimpleModel(nn.Module):
    def __init__(self, num_buckets, embedding_dim):
        super(SimpleModel, self).__init__()
        self.num_buckets = num_buckets
        self.embedding = nn.Embedding(num_buckets, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)  # Example for binary classification

    def forward(self, x):
        # Hash the input to get the corresponding bucket
        hashed_x = torch.tensor([hash_function(v.item(), self.num_buckets) for v in x], dtype=torch.long)
        embed = self.embedding(hashed_x)
        output = self.fc(embed)
        return torch.sigmoid(output)  # Example for binary classification

# Sample data
categorical_data = torch.tensor([12345, 67890, 54321, 98765], dtype=torch.long)  # Example categorical feature
labels = torch.tensor([1, 0, 1, 0], dtype=torch.float32)  # Example labels

# Hyperparameters
num_buckets = 1000  # Number of buckets for hashing
embedding_dim = 8  # Size of the embedding vector

# Initialize model, loss function, and optimizer
model = SimpleModel(num_buckets, embedding_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    outputs = model(categorical_data)
    loss = criterion(outputs.squeeze(), labels)
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')


# Save the trained model
torch.save(model.state_dict(), "simple_model.pth")

# Load the model
model = SimpleModel(num_buckets=num_buckets, embedding_dim=embedding_dim)

# Load the trained model weights
model.load_state_dict(torch.load("simple_model.pth"))

model.eval()

# List of user IDs to inspect
user_ids = [100001, 500500, 750750, 999999, 123456]

# Hash the user IDs to find their bucket indices
bucket_indices = [hash_function(user_id, num_buckets) for user_id in user_ids]

# Convert the bucket indices to a tensor
bucket_indices_tensor = torch.tensor(bucket_indices, dtype=torch.long)

# Retrieve the embeddings from the loaded model's embedding layer
embeddings = model.embedding(bucket_indices_tensor)

# Display the embeddings for each user ID
for i, user_id in enumerate(user_ids):
    print(f"User ID: {user_id}, Bucket Index: {bucket_indices[i]}, Embedding: {embeddings[i]}")




