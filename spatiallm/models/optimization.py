# spatiallm/models/optimization.py

"""
Optimization utilities for SpatialLM training.
"""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Optional, Dict, Any

class AdamW(Optimizer):
    """
    Implements AdamW algorithm with weight decay fix.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                       weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if group['amsgrad']:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Perform weight decay
                p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

                # Perform step
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class WarmupCosineScheduler(_LRScheduler):
    """
    Linear warmup followed by cosine decay learning rate scheduler.
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = float(self.last_epoch - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in self.base_lrs]

class LinearWarmupScheduler(_LRScheduler):
    """
    Linear warmup learning rate scheduler.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            return self.base_lrs

def create_optimizer(model, config):
    """
    Create optimizer based on configuration.
    
    Args:
        model: The model to optimize
        config: Training configuration
        
    Returns:
        Optimizer instance
    """
    # Separate parameters with and without weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'bias' in name or 'LayerNorm' in name or 'layer_norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    # Create parameter groups
    param_groups = [
        {'params': decay_params, 'weight_decay': config.training.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    # Create optimizer
    if config.training.optimizer.type == 'adamw':
        optimizer = AdamW(
            param_groups,
            lr=config.training.learning_rate,
            betas=config.training.optimizer.betas,
            eps=config.training.optimizer.eps,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer.type == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=config.training.learning_rate,
            betas=config.training.optimizer.betas,
            eps=config.training.optimizer.eps
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.training.optimizer.type}")
    
    return optimizer

def create_scheduler(optimizer, config, num_training_steps):
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: The optimizer
        config: Training configuration
        num_training_steps: Total number of training steps
        
    Returns:
        Scheduler instance
    """
    warmup_steps = config.training.warmup_steps
    
    if config.training.scheduler.type == 'cosine':
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=num_training_steps,
            min_lr=config.training.scheduler.get('min_lr', 0)
        )
    elif config.training.scheduler.type == 'linear':
        scheduler = LinearWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps
        )
    else:
        from transformers import get_scheduler
        scheduler = get_scheduler(
            config.training.scheduler.type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    
    return scheduler

class GradientAccumulator:
    """
    Handles gradient accumulation for larger effective batch sizes.
    """
    def __init__(self, accumulation_steps=1):
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
        
    def should_step(self):
        """Check if optimizer should step."""
        self.step_count += 1
        if self.step_count % self.accumulation_steps == 0:
            return True
        return False
        
    def reset(self):
        """Reset step counter."""
        self.step_count = 0

def clip_grad_norm(model, max_norm):
    """
    Clip gradients by norm.
    
    Args:
        model: The model
        max_norm: Maximum gradient norm
        
    Returns:
        Total gradient norm before clipping
    """
    if max_norm <= 0:
        return 0.0
        
    parameters = [p for p in model.parameters() if p.grad is not None]
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    
    return total_norm.item()
