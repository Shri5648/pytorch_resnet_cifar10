import torch
import torch.nn.functional as F
from torch import Tensor

@torch.compile
def SVD_exact(G: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # Compute full SVD of the gradient tensor
    U, S, Vh = torch.linalg.svd(G, full_matrices=True)
    return U, S, Vh

class Muon_pnorm(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, p=2.0):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,p=p)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            p_exp = group['p']
            for p_tensor in group['params']:
                if p_tensor.grad is None:
                    continue
                G = p_tensor.grad
                state = self.state[p_tensor]
                
                # Initialize momentum buffer and step counter
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(G)
                    state['step'] = 0
                
                momentum_buffer = state['momentum_buffer']
                
                # Weight decay (in-place)
                p_tensor.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Momentum update
                momentum_buffer.lerp_(G, 1 - group['momentum'])
                blended_grad = G.lerp(momentum_buffer, group['momentum'])
                
                # Compute SVD of blended gradient
                U, S, Vh = SVD_exact(blended_grad)
                
                # Compute modified singular values: S^{1/p - 1}
                print("p=',p_exp);
                exp = 1.0 / p_exp - 1.0
                S_mod = torch.pow(S, exp)
                
                # Reconstruct search direction D = -U · diag(S_mod) · Vh
                D = -U.matmul(torch.diag(S_mod)).matmul(Vh)
                
                # Update parameters along custom SVD direction
                p_tensor.add_(D, alpha=group['lr'])
                
                # Increment step counter
                state['step'] += 1
