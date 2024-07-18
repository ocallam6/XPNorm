import torch
import torch.nn as nn
import torch.nn.functional as F

class Extinction_Learn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Extinction_Learn, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden layer to output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation function to the output of the first fully connected layer
        x = self.fc2(x)  # Output layer, no activation function
        return x

# Example usage:
# Create an instance of the neural network
input_size = 10
hidden_size = 20
output_size = 1
model = BasicNN(input_size, hidden_size, output_size)

# Generate some dummy input data
dummy_input = torch.randn(5, input_size)  # Batch size of 5, input size of 10

# Pass the input through the model
output = model(dummy_input)
print("Output shape:", output.shape)  # Output shape will be (5, 1) because of batch size 5 and output size 1
