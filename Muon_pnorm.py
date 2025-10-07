import torch
import torch.nn.functional as F
from torch import Tensor

@torch.compile
def SVD_exact(G: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # Compute full SVD of the gradient tensor
    U, S, Vh = torch.linalg.svd(G, full_matrices=True)
    return U, S, Vh

class Muon_pnorm(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                G = p.grad
                state = self.state[p]
                
                # Initialize momentum buffer if first step
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(G)
                    state['step'] = 0
                
                momentum_buffer = state['momentum_buffer']
                
                # Weight decay (in-place)
                p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Momentum update
                momentum_buffer.lerp_(G, 1 - group['momentum'])
                blended_grad = G.lerp(momentum_buffer, group['momentum'])
                
                # Compute SVD of blended gradient
                U, S, Vh = SVD_exact(blended_grad)
                
                # Reconstruct search direction using full SVD:
                # D = U @ diag(S) @ Vh
                D = U.matmul(torch.diag(S)).matmul(Vh)
                
                # Update parameters along SVD direction
                p.add_(D, alpha=-group['lr'])
                
                # Increment step counter
                state['step'] += 1
