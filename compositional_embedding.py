import torch
import torch.nn as nn
import torch.optim as optim

class ComplementaryPartitionEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, partition_sizes, operation="concat"):
        super(ComplementaryPartitionEmbedding, self).__init__()
        # List containing sizes for each partition
        self.partition_sizes = partition_sizes
        # Total number of users
        self.num_embeddings = num_embeddings
        self.operation = operation

        # Check if the product of partition sizes can accommodate the number of users
        product_of_partitions = 1
        for size in partition_sizes:
            product_of_partitions *= size
        assert self.num_embeddings <= product_of_partitions, "The number of users must be <= product of partition sizes."

        # Set partition embedding dimension
        if operation == "concat":
            # Split embedding dimension for concatenation
            # we have to pay extra attention in this part as we want
            # the concatenation of all partions to be equal to embedding_dim
            # e.g., with embedding_dim=16, and len(partition_sizes)=3, we
            # do not get the actual embedding_dim after concatenation
            self.partition_dim = embedding_dim // len(partition_sizes)
        else:
            # Full embedding dimension for sum/multiply
            self.partition_dim = embedding_dim

        # Create embedding tables for each partition
        self.sub_embeddings = nn.ModuleList([
            nn.Embedding(partition_size, self.partition_dim) for partition_size in partition_sizes
        ])

    def partition_user_ids(self, user_ids):
        """
        Assigns each user_id to an equivalence class in each partition.
        """
        partition_indices = []
        for partition_size in self.partition_sizes:
            # Each user_id is mapped to a class in the partition
            partition_indices.append(user_ids % partition_size)

        return partition_indices

    def forward(self, user_ids):
        # Get partition indices for each user ID
        partition_indices = self.partition_user_ids(user_ids)

        # Collect embeddings from each partition
        embeddings = []
        for sub_emb, partition_idx in zip(self.sub_embeddings, partition_indices):
            embeddings.append(sub_emb(partition_idx))

        # Combine embeddings based on the specified operation
        if self.operation == "concat":
            composed_embedding = torch.cat(embeddings, dim=-1)
        elif self.operation == "sum":
            composed_embedding = torch.sum(torch.stack(embeddings), dim=0)
        elif self.operation == "multiply":
            composed_embedding = torch.prod(torch.stack(embeddings), dim=0)

        return composed_embedding

# Example usage for user IDs
# The number of users must be less than or equal to the product of partition sizes (20 * 10 * 8 = 1600)
num_users = 1000
# Original embedding dimension
embedding_dim = 16
# Sizes of each partition
partition_sizes = [20, 10, 5, 3]


class SimpleRecSys(nn.Module):
    def __init__(self, num_users, embedding_dim, partition_sizes, operation="concat"):
        super(SimpleRecSys, self).__init__()
        # User embedding based on complementary partitioning
        self.user_embedding = ComplementaryPartitionEmbedding(num_users, embedding_dim, partition_sizes, operation)

        # A simple linear layer to predict the score from the user embedding
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, user_ids):
        # Get user embedding
        user_emb = self.user_embedding(user_ids)

        # Predict score using the linear layer
        scores = self.fc(user_emb)
        return scores


# Dummy dataset
# Example user IDs
user_ids = torch.tensor([1, 5, 10, 50, 100, 200])
# Example interaction scores (e.g., rating)
interaction_scores = torch.tensor([1.0, 0.5, 0.8, 1.0, 0.3, 0.7])

# Instantiate the recommendation system
model = SimpleRecSys(num_users, embedding_dim, partition_sizes, "concat")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()

    # Forward pass
    predictions = model(user_ids)
    loss = criterion(predictions.squeeze(), interaction_scores)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss for every 10th epoch
    if epoch % 10 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

print("Training completed!")

# Inference for a test user (e.g., user ID 10)
test_user_id = torch.tensor([10])

# Set the model to evaluation mode
model.eval()

# Get user embedding and predicted score
with torch.no_grad():
    user_embedding = model.user_embedding(test_user_id)
    predicted_score = model(test_user_id)

print(f"User Embedding for User ID {test_user_id.item()}: {user_embedding}")
print(f"Predicted Score for User ID {test_user_id.item()}: {predicted_score.item()}")