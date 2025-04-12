import torch
#text = "There is an obstaclhe ahead."
#if "obstacle" in text:
#    print("Found!")
#
#print(torch.randn(0,3))


x = torch.tensor([[1, 2, 9], [3, 4, 7]])
print(x.size())
x = x.unsqueeze(1)
print(x)
print(x.size())

