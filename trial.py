import torch, numpy as np
def load(path): return torch.load(path, weights_only=False)['graphs']
for split in ['sim_train.pt','sim_val.pt']:
    gs = load(f'datasets/{split}')
    y = np.array([int(g.y.item()) for g in gs])
    pos = y.mean()
    print(split, 'pos_rate=', pos, 'majority_acc=', 1-pos)
