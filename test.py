import sparsenn
import torch

for hdim in range(2, 7):
    l = sparsenn.HypercubeLinear(in_features=2**hdim, out_features=2**hdim, hdim=hdim)
    assert (l.weight() != 0.0).sum().item() == (hdim+1) * 2**hdim
    

# for s1 in range(2, 6):
    # for s2 in range(2, 6):
        # l = sparsenn.KroneckerLinear(
            # in_features=s1*(s1+1), out_features=s2*(s2+1),
            # kronecker_shapes=[(s1, s2), (s1+1, s2+1)])
        # assert l.weight().shape == (s1*(s1+1)) * (s2*(s2+1))