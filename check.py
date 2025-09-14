import torch
train = torch.load("datasets/sim_train.pt", weights_only=False)["graphs"]
val   = torch.load("datasets/sim_val.pt",   weights_only=False)["graphs"]
test  = torch.load("datasets/sim_test.pt",  weights_only=False)["graphs"]
print("graphs:", len(train), len(val), len(test))
g0 = train
print("x:", g0.x.shape, "edge_index:", g0.edge_index.shape, "y:", g0.y.shape)  # expect (N,9),(2,E),(1,)
g0.validate(raise_on_error=True)
