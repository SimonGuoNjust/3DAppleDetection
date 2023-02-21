import torch
model = torch.load('best.pt')
print(model)
torch.save(model['model'].state_dict(),'best_state_dict.pt')