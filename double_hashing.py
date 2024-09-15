import torch
import torch.nn as nn
import torch.optim as optim
import hashlib


class DoubleHashingEmbedding(nn.Module):
    def __init__(self, num_buckets, embedding_dim, aggregation="sum"):
        super(DoubleHashingEmbedding, self).__init__()
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim
        self.aggregation = aggregation

        # Single embedding layer
        self.embedding = nn.Embedding(num_buckets, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def hash_function_1(self, f):
        """Applies the first hash function (h1) using MD5."""
        return int(hashlib.md5(f.encode()).hexdigest(), 16) % self.num_buckets

    def hash_function_2(self, f):
        """Applies the second hash function (h2) using SHA1."""
        return int(hashlib.sha1(f.encode()).hexdigest(), 16) % self.num_buckets

    def forward(self, features):
        """Forward pass to get the embedding for a batch of features."""
        # Ensure that features are strings and handle batching
        if isinstance(features, torch.Tensor):
            features = features.tolist()  # Convert tensor to list

        # Compute hashes for the batch of features
        h1 = [self.hash_function_1(str(f)) for f in features]
        h2 = [self.hash_function_2(str(f)) for f in features]

        # Convert hash indices to tensors
        h1_tensor = torch.tensor(h1, dtype=torch.long, device=self.embedding.weight.device)
        h2_tensor = torch.tensor(h2, dtype=torch.long, device=self.embedding.weight.device)

        # Get embeddings corresponding to both hashes
        E_h1 = self.embedding(h1_tensor)
        E_h2 = self.embedding(h2_tensor)

        # Aggregate embeddings based on the chosen method
        if self.aggregation == "sum":
            return E_h1 + E_h2
        elif self.aggregation == "concat":
            return torch.cat([E_h1, E_h2], dim=-1)
        elif self.aggregation == "multiplication":
            return E_h1 * E_h2
        else:
            raise ValueError("Invalid aggregation method. Choose 'sum', 'concat', or 'multiplication'.")


class SimpleRecSysModel(nn.Module):
    def __init__(self, num_buckets, embedding_dim, output_dim):
        super(SimpleRecSysModel, self).__init__()
        self.double_hash_embedding = DoubleHashingEmbedding(num_buckets, embedding_dim)
        if self.double_hash_embedding.aggregation == "concat":
            input_dim = embedding_dim * 2
        else:
            input_dim = embedding_dim
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, features):
        embeddings = self.double_hash_embedding(features)
        output = self.fc(embeddings)
        return output


# Configuration
num_buckets = 100
embedding_dim = 16
output_dim = 1
epochs = 10

# Sample data: User IDs
user_ids = torch.tensor([13523, 234456, 784549, 34535], dtype=torch.long)
# Labels for binary classification
labels = torch.tensor([1, 1, 0, 0], dtype=torch.float32)

# Model, Loss, Optimizer
model = SimpleRecSysModel(num_buckets, embedding_dim, output_dim)
# For binary classification
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    # Ensure shape matches for BCEWithLogitsLoss
    outputs = model(user_ids).squeeze()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


# Function to get embedding for a specific user_id
def get_embedding_for_user(model, user_id):
    model.eval()
    with torch.no_grad():
        embedding = model.double_hash_embedding(torch.tensor([user_id], dtype=torch.long))
    return embedding


# Example test user_id
test_user_id = 23435
embedding_vector = get_embedding_for_user(model, test_user_id)
print(f"Embedding vector for userId {test_user_id}: {embedding_vector}")
