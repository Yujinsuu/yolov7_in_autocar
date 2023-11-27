import torch

# Load the model checkpoint
model = torch.load('best.pt')['model']

# Extract the state dict and remove the 'module' prefix
state_dict = model.state_dict()
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# Save the state dict as a weights file
torch.save(state_dict, 'weights/yolov7.weights')