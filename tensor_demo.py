import torch

a = torch.tensor([1, 2, 3])
print(a)

b = torch.rand(3, 3)
print(b)

c = a + 1
print(c)

d = torch.matmul(b, b)
print(d)
