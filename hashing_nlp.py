from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import hashlib

## Download NLTK data (run these lines once)
## for tokenization
# nltk.download('punkt')
## for stop word removal
# nltk.download('stopwords')

# Sample documents
documents = [
    "In the hashing trick, input features are mapped into a smaller dimensional space using a hashing function.",
    "This approach is a form of dimensionality reduction, which helps manage memory in large datasets.",
    "It also helps encountering new tokens.",
    "However, this can result in collisions, where different features are assigned the same hash."
]

# Function to preprocess the document
def pre_process(text):
    # Initialize the stemmer
    stemmer = PorterStemmer()

    # Lowercasing
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword Removal
    stop_words = set(stopwords.words('english'))

    tokens = [word for word in tokens if word not in stop_words]

    # Punctuation Removal
    tokens = [word for word in tokens if word.isalnum()]

    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]

    return tokens

# Apply preprocessing to each document
processed_documents = [pre_process(doc) for doc in documents]

# Check the results
for i, doc in enumerate(processed_documents):
    print(f"Document {i+1}: {doc}")


# Hash function to compute the sign by appending "_sign" to the token
def sign_hash(token, hash_algorithm='SHA1'):
    # Generate a hash value for the token with the '_sign' suffix
    hash_value = token_hash(token + "_sign", hash_algorithm)

    # Determine the sign based on the modulus operation
    return 1 if hash_value % 2 == 0 else -1

# Hash function to compute a unique hash for the token itself
def token_hash(token, hash_algorithm='SHA1'):
    # Create a hash object using the specified algorithm
    hash_obj = hashlib.new(hash_algorithm)

    # Update the hash object with the token encoded as bytes
    hash_obj.update(token.encode('utf-8'))

    # Get the hexadecimal digest of the hash
    hash_value_hex = hash_obj.hexdigest()

    # Convert the hexadecimal hash value to a decimal integer
    hash_value_dec = int(hash_value_hex, 16)

    # Return the decimal hash value
    return hash_value_dec

# Function to apply the hashing trick with a fixed hash size (5 in this case)
def document_hash_vector(doc, hash_size=5):
    # Initialize a vector of size 5
    vector = [0] * hash_size

    for token in doc:
        # Hash the token
        hash_value = token_hash(token)

        # Get the sign (+1 or -1)
        sign = sign_hash(token)

        # Map the hash value to an index in the vector (0 to 4)
        index = hash_value % hash_size

        # Update the vector at the index
        vector[index] += sign

    return vector

# Apply the hashing trick
hashed_documents = []
for doc in processed_documents:
    hashed_doc = [(token, token_hash(token), sign_hash(token)) for token in doc]
    hashed_documents.append(hashed_doc)

# Print the results
for i, doc in enumerate(hashed_documents):
    print(f"Document {i+1} Hash Values and Signs:")
    for token, hash_value, sign in doc:
        print(f"Token: {token}, Hash Value: {hash_value}, Sign: {sign}")
    print("\n")


# Compute the final vector for each document
final_vectors = [document_hash_vector(doc) for doc in processed_documents]

# Print the results
for i, vector in enumerate(final_vectors):
    print(f"Document {i+1} Final Vector: {vector}")