import torch
import torch.nn as nn
import torch.optim as optim
import hashlib


class SingleEmbeddingDoubleHashing(nn.Module):
    def __init__(self, B, embedding_dim, aggregation="sum"):
        super(SingleEmbeddingDoubleHashing, self).__init__()
        self.B = B
        self.embedding_dim = embedding_dim
        self.aggregation = aggregation

        # Single embedding layer
        self.embedding = nn.Embedding(B, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def hash_function_1(self, f):
        """Applies the first hash function (h1) using MD5."""
        return int(hashlib.md5(f.encode()).hexdigest(), 16) % self.B

    def hash_function_2(self, f):
        """Applies the second hash function (h2) using SHA1."""
        return int(hashlib.sha1(f.encode()).hexdigest(), 16) % self.B

    def forward(self, features):
        """Forward pass to get the embedding for the list of features."""
        embeddings = []
        for f in features:
            h1 = self.hash_function_1(f)
            h2 = self.hash_function_2(f)

            E_h1 = self.embedding(torch.tensor(h1, dtype=torch.long))
            E_h2 = self.embedding(torch.tensor(h2, dtype=torch.long))

            if self.aggregation == "sum":
                embeddings.append(E_h1 + E_h2)
            elif self.aggregation == "concat":
                embeddings.append(torch.cat([E_h1, E_h2], dim=-1))
            else:
                raise ValueError("Invalid aggregation method. Choose either 'sum' or 'concat'.")

        # Stack embeddings to form a batch tensor
        return torch.stack(embeddings)


class SimpleModel(nn.Module):
    def __init__(self, B, embedding_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.double_hash_embedding = SingleEmbeddingDoubleHashing(B, embedding_dim)
        input_dim = embedding_dim if self.double_hash_embedding.aggregation == "sum" else embedding_dim * 2
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, features):
        embedding = self.double_hash_embedding(features)
        output = self.fc(embedding)
        return output


# Training setup
def train_model(model, data_loader, epochs=1):
    criterion = nn.CrossEntropyLoss()  # Assuming a classification problem
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        for features, labels in data_loader:
            # Extract feature strings
            feature_strings = [feature for feature in features]
            optimizer.zero_grad()
            outputs = model(feature_strings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


# Example DataLoader (dummy data for illustration)
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, B, num_samples):
        self.B = B
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        feature = f"user_{idx % self.B}"
        label = idx % 10  # Dummy label
        return feature, label


B = 100  # Base hash space size
embedding_dim = 16  # Embedding dimension
output_dim = 10  # Number of classes or output units for the model
num_samples = 1000  # Number of samples in the dataset

# Instantiate the model
model = SimpleModel(B, embedding_dim, output_dim)

# Create a dummy DataLoader for illustration
data_loader = torch.utils.data.DataLoader(DummyDataset(B, num_samples), batch_size=32, shuffle=True)

# Train the model
train_model(model, data_loader, epochs=2)


# Extract the embedding vector for a test user_id
def get_embedding_for_user(model, user_id):
    model.eval()
    with torch.no_grad():
        embedding = model.double_hash_embedding([user_id])  # Pass as list
    return embedding


# Example test user_id
test_user_id = "user_12345"
embedding_vector = get_embedding_for_user(model, test_user_id)
print(f"Embedding vector for {test_user_id}: {embedding_vector}")
