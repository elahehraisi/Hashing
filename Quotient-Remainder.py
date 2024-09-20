import torch
import torch.nn as nn


class QuotientRemainderEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, M):
        super(QuotientRemainderEmbedding, self).__init__()
        # partition size
        self.M = M
        self.quotient_embeddings = nn.Embedding(num_embeddings // M + 1, embedding_dim)
        self.remainder_embeddings = nn.Embedding(M, embedding_dim)

    def forward(self, x):
        # Calculate quotient and remainder
        quotient = x // self.M
        remainder = x % self.M

        # Get embeddings for quotient and remainder
        emb_q = self.quotient_embeddings(quotient)
        emb_r = self.remainder_embeddings(remainder)

        # Perform element-wise multiplication instead of concatenation
        composed_emb = emb_q * emb_r

        return composed_emb


# Define a simple model with QuotientRemainderEmbedding
class RecommendationModel(nn.Module):
    def __init__(self, num_users, embedding_dim, M):
        super(RecommendationModel, self).__init__()
        self.user_embedding = QuotientRemainderEmbedding(num_users, embedding_dim, M)
        # Final layer to predict the score
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, user_ids):
        user_emb = self.user_embedding(user_ids)
        # Predict score for each user
        output = self.fc(user_emb)
        return output



# For example, assume 10,000 users
num_users = 10000
embedding_dim = 16
# Partition size for Quotient and Remainder
M = 100
learning_rate = 0.001
num_epochs = 5

# Initialize the model
model = RecommendationModel(num_users, embedding_dim, M)
# Loss function for regression
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Example training data: User IDs and corresponding scores (randomly generated for the demo)
# Batch of 32 user IDs
user_ids = torch.randint(0, num_users, (32,))
true_scores = torch.rand(32, 1)

# Training loop
for epoch in range(num_epochs):
    model.train()

    # Forward pass: Compute predicted score by passing user_ids through the model
    predicted_scores = model(user_ids)

    # Compute the loss
    loss = criterion(predicted_scores, true_scores)

    # Backward pass: Compute gradients
    optimizer.zero_grad()
    loss.backward()

    # Update the weights
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# Now let's retrieve the embedding for a test userId
test_user_id = torch.tensor([123])

# Extract the embedding
with torch.no_grad():
    test_embedding = model.user_embedding(test_user_id)

print(f"Embedding for test userId 123: {test_embedding}")