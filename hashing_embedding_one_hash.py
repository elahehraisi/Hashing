import torch
import torch.nn as nn
import torch.optim as optim
import hashlib


def hash_function(value, num_buckets):
    # Hash value using hashlib and take the modulus to map to the bucket space
    hash_value = int(hashlib.md5(str(value).encode('utf-8')).hexdigest(), 16)
    return hash_value % num_buckets

class HashedEmbedding(nn.Module):
    def __init__(self, num_buckets, embedding_dim):
        super(HashedEmbedding, self).__init__()
        self.num_buckets = num_buckets
        self.embedding = nn.Embedding(num_buckets, embedding_dim)
        # for binary classification
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # Hash the input to get the corresponding bucket index
        hashed_x = torch.tensor([hash_function(v.item(), self.num_buckets) for v in x], dtype=torch.long)

        # The embedding for the hashed bucket
        embed = self.embedding(hashed_x)
        output = self.fc(embed)
        return torch.sigmoid(output)

# Sample data for user id
categorical_data = torch.tensor([34534, 464677, 77477, 98765, 93872, 13243], dtype=torch.long)
# Example labels
labels = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.float32)

# Number of buckets for hashing
num_buckets = 100

# Size of the embedding vector
embedding_dim = 8

# Initialize model, loss function, and optimizer
model = HashedEmbedding(num_buckets, embedding_dim)
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
torch.save(model.state_dict(), "HashedEmbedding.pth")

# Set up the model
model = HashedEmbedding(num_buckets=num_buckets, embedding_dim=embedding_dim)

# Load the model weights
model.load_state_dict(torch.load("HashedEmbedding.pth"))

model.eval()
# List of user IDs to inspect
user_ids = [100001, 500500]

# Hash the user IDs to find their bucket indices
bucket_indices = [hash_function(user_id, num_buckets) for user_id in user_ids]

# Convert the bucket indices to a tensor
bucket_indices_tensor = torch.tensor(bucket_indices, dtype=torch.long)

# Retrieve the embeddings from the loaded model's embedding layer
embeddings = model.embedding(bucket_indices_tensor)

# Display the embeddings for each user ID
for i, user_id in enumerate(user_ids):
    print(f"User ID: {user_id}, Bucket Index: {bucket_indices[i]}, Embedding: {embeddings[i]}")




