import torch
import torch.nn as nn
import torch.optim as optim
import hashlib
import random


# Step 1: Define the hashing function and hash vector generation
def hash_function(value, seed):
    """Hash function that uses md5 and a seed for randomness."""
    return int(hashlib.md5(f"{value}_{seed}".encode('utf8')).hexdigest(), 16) % 100000  # Keep hash values manageable


def get_hashed_vector(value, num_hash_functions):
    """Generate a vector of hashed values."""
    return torch.tensor([hash_function(value, seed) for seed in range(num_hash_functions)])


# Step 2: Define the Deep Hash Encoder (DHE) model
class DeepHashEncoder(nn.Module):
    def __init__(self, num_hash_functions, output_dim):
        super(DeepHashEncoder, self).__init__()
        self.fc1 = nn.Linear(num_hash_functions, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, hash_vector):
        x = torch.relu(self.fc1(hash_vector.float()))
        x = self.fc2(x)
        return x


# Step 3: Define a simple training loop for user embedding
class UserEmbeddingModel(nn.Module):
    def __init__(self, num_hash_functions, embedding_dim):
        super(UserEmbeddingModel, self).__init__()
        self.user_encoder = DeepHashEncoder(num_hash_functions, embedding_dim)

    def forward(self, user_hash_vector):
        user_embedding = self.user_encoder(user_hash_vector)
        return user_embedding


# Sample data
user_ids = list(range(1, 1001))  # Simulating 1000 unique user IDs
target_values = [random.random() for _ in range(1000)]  # Random target values for training

# Step 4: Instantiate the model and optimizer
num_hash_functions = 4
embedding_dim = 16

model = UserEmbeddingModel(num_hash_functions=num_hash_functions, embedding_dim=embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

# Step 5: Training the model
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for user_id, target in zip(user_ids, target_values):
        # Generate hash vector for the user ID
        user_hash_vector = get_hashed_vector(user_id, num_hash_functions)

        # Forward pass
        optimizer.zero_grad()
        user_embedding = model(user_hash_vector)

        # Simulated loss with random target values
        target_tensor = torch.tensor([target], dtype=torch.float)
        loss = loss_function(user_embedding.sum(), target_tensor)  # Simplified loss function

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(user_ids):.4f}")

# Step 6: Generating embedding for a new user ID
new_user_id = 54321  # Example of a new user ID
new_user_hash_vector = get_hashed_vector(new_user_id, num_hash_functions)

# Get the embedding from the trained model
new_user_embedding = model(new_user_hash_vector)

print(f"Embedding for user_id {new_user_id}:")
print(new_user_embedding)
