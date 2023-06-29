import torch
import matplotlib.pyplot as plt

derivatives = 3

X = torch.linspace(0, torch.pi/3, 100, dtype=torch.float64).reshape((-1,1))
X.requires_grad = True
y = torch.sin(X) + torch.pow(X, torch.divide(10, X))

dy_dx = [torch.autograd.grad(y, X, grad_outputs=torch.ones_like(X),
                create_graph=True)[0]]

print(f"(df_dx)^{1} = {dy_dx[0]}")
for i in range(derivatives-1):
    import pdb;pdb.set_trace()
    dy_dx.append(torch.autograd.grad(dy_dx[-1], X, grad_outputs=torch.ones_like(X),
                    create_graph=True)[0])
    print(f"(df_dx)^{i+2} = {dy_dx[i+1]}")

fig, ax = plt.subplots()

ax.plot(X.detach().numpy(), y.detach().numpy(), label="f(X)")
for i, dy_dx in enumerate(dy_dx):
    ax.plot(X.detach().numpy(), dy_dx.detach().numpy(), label=f"(df(X)/dX)^{i+1}")
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.legend()
plt.show()
import pdb;pdb.set_trace()

