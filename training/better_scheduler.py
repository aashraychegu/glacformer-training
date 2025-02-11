import math
from torch.optim.lr_scheduler import _LRScheduler
# C osine
# A nnealing
# W arm 
# R estart
# W arm 
# U p
# D ecaying
# P eaks
class CosineAnnealingWarmRestartWarmUpDecayingPeaks(_LRScheduler):
    """
    Combines cosine annealing with warm restarts, warmup periods, and gamma decay.
    Includes warmup periods before each restart.
    
    Args:
        optimizer: Wrapped optimizer
        first_cycle_steps: Length of the first cycle in steps (including warmup)
        cycle_mult: Multiplicative factor for cycle length after each restart
        max_lr: Upper learning rate boundary in the cycle
        min_lr: Lower learning rate boundary in the cycle
        warmup_steps: Number of warmup steps at the start of each cycle
        gamma: Multiplicative factor for max_lr after each restart
        verbose: If True, prints information about learning rate updates
        last_epoch: The index of the last epoch
    """
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1.0, max_lr=1.0,
                 min_lr=1e-6, warmup_steps=0, gamma=1.0, verbose=False, last_epoch=-1):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.verbose = verbose
        
        self.cur_cycle_steps = first_cycle_steps
        self.step_in_cycle = last_epoch
        self.cycle = 0
        self.in_warmup = True  # Track if we're in warmup phase
        self.total_steps = 0  # Track total steps taken
        
        super(CosineAnnealingWarmRestartWarmUpDecayingPeaks, self).__init__(optimizer, last_epoch)
        
        # Initialize learning rate to min_lr
        self.init_lr()
        
        if self.verbose:
            print("\nInitialized scheduler with:")
            print(f"  First cycle steps: {first_cycle_steps}")
            print(f"  Warmup steps: {warmup_steps}")
            print(f"  Max LR: {max_lr}")
            print(f"  Min LR: {min_lr}")
            print(f"  Cycle multiplier: {cycle_mult}")
            print(f"  Gamma decay: {gamma}\n")
    
    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            
        if self.verbose:
            print(f"Set initial learning rate to {self.min_lr}")
    
    def print_lr(self, lr):
        """Print current learning rate with detailed information"""
        phase = "warmup" if self.in_warmup else "cosine"
        print(f"Step {self.total_steps} | Epoch {self.last_epoch} | "
              f"Cycle {self.cycle} ({self.step_in_cycle}/{self.cur_cycle_steps}) | "
              f"Phase: {phase} | LR: {lr:.6f}")
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return [self.min_lr for _ in self.base_lrs]
        
        # Handle warmup period
        if self.in_warmup and self.step_in_cycle < self.warmup_steps:
            # Linear warmup
            lr_scale = self.step_in_cycle / self.warmup_steps
            current_lr = (self.max_lr - self.min_lr) * lr_scale + self.min_lr
            
            if self.verbose:
                self.print_lr(current_lr)
            
            return [current_lr for _ in self.base_lrs]
        
        # Calculate progress in the cosine annealing period
        cosine_steps = self.cur_cycle_steps - self.warmup_steps
        cosine_progress = (self.step_in_cycle - self.warmup_steps) / cosine_steps
        
        # Check if we need to restart
        if cosine_progress >= 1.0:
            # Time for a restart - begin new warmup
            self.cycle += 1
            self.step_in_cycle = 0
            old_cycle_steps = self.cur_cycle_steps
            self.cur_cycle_steps = int(self.cur_cycle_steps * self.cycle_mult)
            old_max_lr = self.max_lr
            self.max_lr *= self.gamma
            self.in_warmup = True
            
            if self.verbose:
                print(f"\nStarting cycle {self.cycle}:")
                print(f"  New cycle length: {old_cycle_steps} -> {self.cur_cycle_steps}")
                print(f"  New max LR: {old_max_lr:.6f} -> {self.max_lr:.6f}\n")
            
            return [self.min_lr for _ in self.base_lrs]
        
        # If we've finished warmup, switch to cosine annealing
        if self.in_warmup and self.step_in_cycle >= self.warmup_steps:
            self.in_warmup = False
            if self.verbose:
                print(f"\nFinished warmup, starting cosine annealing for cycle {self.cycle}\n")
        
        # Calculate cosine annealing
        cosine_val = 0.5 * (1 + math.cos(math.pi * cosine_progress))
        current_lr = (self.max_lr - self.min_lr) * cosine_val + self.min_lr
        
        if self.verbose:
            self.print_lr(current_lr)
        
        return [current_lr for _ in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            self.step_in_cycle += 1
            self.total_steps += 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = 0
                self.cur_cycle_steps = int(self.cur_cycle_steps * self.cycle_mult)
                self.max_lr *= self.gamma
                self.in_warmup = True  # Start new warmup period
        else:
            self.step_in_cycle = epoch
        super(CosineAnnealingWarmRestartWarmUpDecayingPeaks, self).step()

better_scheduler = CosineAnnealingWarmRestartWarmUpDecayingPeaks