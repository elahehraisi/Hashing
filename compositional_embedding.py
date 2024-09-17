import torch
import torch.nn as nn
import torch.optim as optim


class CompositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_partitions):
        super(CompositionalEmbedding, self).__init__()
        self.num_partitions = num_partitions
        self.partition_dim = embedding_dim // num_partitions

        # Create sub-embedding tables for each partition
        self.sub_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, self.partition_dim) for _ in range(num_partitions)
        ])

    def forward(self, indices):
        # Collect sub-embeddings from each partition
        embeddings = [sub_emb(indices) for sub_emb in self.sub_embeddings]

        # Perform element-wise multiplication instead of concatenation
        composed_embedding = embeddings[0]
        for emb in embeddings[1:]:
            # Element-wise multiplication
            composed_embedding = composed_embedding * emb

        return composed_embedding


class SimpleModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_partitions, output_dim=1):
        super(SimpleModel, self).__init__()
        self.embedding_layer = CompositionalEmbedding(num_embeddings, embedding_dim, num_partitions)

        # A simple feed-forward network after the embedding
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, user_id):
        # Get the compositional embedding for the userId
        embedding = self.embedding_layer(user_id)
        output = self.fc(embedding)
        return output


# Hyperparameters
# Total number of users
num_users = 1000
# Size of the final user embedding
embedding_dim = 32
# Number of partitions to divide the embedding into, take >=3
num_partitions = 4
# Output dimension (for regression or binary classification)
output_dim = 1

# Create model, loss function, and optimizer
model = SimpleModel(num_users, embedding_dim, num_partitions, output_dim)
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Data (Example)
user_ids = torch.randint(0, num_users, (1000,))  # Random user IDs
targets = torch.rand(1000)  # Random target values (e.g., ratings, clicks)

# Training Loop
epochs = 10
for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()

    # Forward pass
    predictions = model(user_ids)

    # Compute loss
    loss = criterion(predictions.squeeze(), targets)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Get embedding for a test userId
test_user_id = torch.tensor([5])  # Example test user ID
model.eval()
with torch.no_grad():
    test_user_embedding = model.embedding_layer(test_user_id)

print("Test User Embedding:", test_user_embedding)
