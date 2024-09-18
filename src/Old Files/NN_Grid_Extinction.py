'''
Here i want to create a neural network which takes in the full file, and learns the map which beest describes extinction.
Conditions:
Should be non-decreasing along radial direction
correlation to different sighlines


Set up as a l,w,d grid
along the d direction is nondecresing in each sighline
also correlated in the l,w of each d bin
can i implement it as a neural network with d layers
at each layer it looks at the l,w grid but not each node in the l,w grid is coonnected, only connected to ones near them, might have to
be padded
at each layer it throws out a value for each l,w grid, and the next layer adds to that. 

The function is constant but the grid itself is the parameters

work forwards and branch off, then minimise the nn for the parameters, the nn foesnt have any inputs

put in all zeros, it computes the ext of the first window and so on 

i need a map A:l,w,d ->stars
some sort of tokenisation

each datapoint can have an index associated with it mapping associated with it


'''
import torch
import torch.optim as optim

l=20
b=20
r=50
# Define your parameter grid
grid_size = (l, b, r)
first_layer=torch.zeros((l, b, 1))
parameters = torch.randn((l, b, r-1), requires_grad=True)
extinction=torch.concatenate([first_layer,parameters],dim=2)
for i in range(1,r):
    extinction[:,:,i]+=extinction[:,:,i-1]


# Define your loss function
def loss_function(predicted_extinction, ground_truth_extinction):
    return torch.mean((predicted_extinction - ground_truth_extinction) ** 2)

# Define your optimizer
optimizer = optim.SGD([parameters], lr=0.01)

# Training loop
for epoch in range(10):
    # Iterate through your data
        # Zero the gradients
    optimizer.zero_grad()
    
    # Compute the predicted extinction using the parameter grid
    
    # Compute the loss
    loss = loss_function(extinction, target_extinction)
    
    # Backpropagation
    loss.backward()
    
    # Update the parameters
    optimizer.step()

    # Print the loss after each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
