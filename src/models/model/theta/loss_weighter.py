"""
Adaptive Loss Weighter for Supervised ETM

自适应损失权重调整器，用于平衡分类损失和重构损失的数值尺度。

注意：KL 散度不参与自适应平衡，以避免后验坍塌问题。
KL 损失应严格遵守原有的 kl_weight 逻辑。
"""

import torch
from typing import Dict, Optional


class AdaptiveLossWeighter:
    """
    自适应损失权重调整器
    
    在 warm-up 阶段统计 CE 和 Recon 损失的移动平均值，
    然后根据损失尺度自动计算权重，使各损失贡献符合目标比例。
    
    重要：KL 散度不参与自适应平衡，以避免 VAE 后验坍塌问题。
    
    Args:
        warmup_steps: Warm-up 步数，在此期间统计损失 EMA
        ema_decay: 指数移动平均衰减系数
        target_ratio: 目标贡献比例，如 {'ce': 0.7, 'recon': 0.3}
        warmup_ce_weight: Warm-up 阶段 CE 的初始权重（防止分类器走偏）
        min_weight: 权重下限
        max_weight: 权重上限
    
    Example:
        >>> weighter = AdaptiveLossWeighter(warmup_steps=100)
        >>> for step, batch in enumerate(dataloader):
        ...     ce_loss, recon_loss, kl_loss = model(batch)
        ...     weights = weighter.update({'ce': ce_loss, 'recon': recon_loss})
        ...     total_loss = weights['ce'] * ce_loss + weights['recon'] * recon_loss + kl_loss
    """
    
    def __init__(
        self,
        warmup_steps: int = 100,
        ema_decay: float = 0.99,
        target_ratio: Optional[Dict[str, float]] = None,
        warmup_ce_weight: float = 10.0,
        min_weight: float = 0.1,
        max_weight: float = 50.0
    ):
        self.warmup_steps = warmup_steps
        self.ema_decay = ema_decay
        self.target_ratio = target_ratio or {'ce': 0.7, 'recon': 0.3}
        self.warmup_ce_weight = warmup_ce_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # 移动平均统计（仅统计 ce 和 recon）
        self.ema_losses: Dict[str, float] = {}
        self.step_count = 0
        self.weights_frozen = False
        self.frozen_weights: Dict[str, float] = {}
        
        # Warm-up 阶段的默认权重
        self.warmup_weights = {
            'ce': warmup_ce_weight,  # 较高初始值，防止分类器被 recon 淹没
            'recon': 1.0
        }
    
    def update(self, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        更新损失统计并返回当前权重
        
        Args:
            loss_dict: {'ce': ce_loss, 'recon': recon_loss}
                       注意：不要传入 kl_loss，KL 不参与自适应平衡
        
        Returns:
            weights: {'ce': w_ce, 'recon': w_recon}
        """
        self.step_count += 1
        
        # 仅更新 ce 和 recon 的 EMA（忽略任何其他损失如 kl）
        for name in ['ce', 'recon']:
            if name not in loss_dict:
                continue
            
            loss = loss_dict[name]
            loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
            
            if name not in self.ema_losses:
                self.ema_losses[name] = loss_val
            else:
                self.ema_losses[name] = (
                    self.ema_decay * self.ema_losses[name] + 
                    (1 - self.ema_decay) * loss_val
                )
        
        # Warm-up 阶段结束后冻结权重
        if self.step_count == self.warmup_steps and not self.weights_frozen:
            self._freeze_weights()
        
        return self.get_weights()
    
    def _freeze_weights(self):
        """根据 EMA 统计计算并冻结权重"""
        if 'ce' not in self.ema_losses or 'recon' not in self.ema_losses:
            print(f"[AdaptiveLossWeighter] Warning: Missing loss statistics, using warmup weights")
            self.frozen_weights = self.warmup_weights.copy()
            self.weights_frozen = True
            return
        
        ema_ce = self.ema_losses['ce']
        ema_recon = self.ema_losses['recon']
        
        if ema_ce < 1e-8 or ema_recon < 1e-8:
            print(f"[AdaptiveLossWeighter] Warning: Loss too small, using warmup weights")
            self.frozen_weights = self.warmup_weights.copy()
            self.weights_frozen = True
            return
        
        # 计算权重使得加权后的损失符合目标比例
        # 目标: w_ce * ema_ce : w_recon * ema_recon = target_ce : target_recon
        # 设 w_recon = 1.0，则 w_ce = (target_ce / target_recon) * (ema_recon / ema_ce)
        target_ce = self.target_ratio['ce']
        target_recon = self.target_ratio['recon']
        
        w_ce = (target_ce / target_recon) * (ema_recon / ema_ce)
        w_ce = max(self.min_weight, min(self.max_weight, w_ce))
        
        self.frozen_weights = {
            'ce': w_ce,
            'recon': 1.0
        }
        self.weights_frozen = True
        
        # 计算加权后的预期贡献比例
        weighted_ce = w_ce * ema_ce
        weighted_recon = 1.0 * ema_recon
        total_weighted = weighted_ce + weighted_recon
        
        print(f"\n{'='*60}")
        print(f"[AdaptiveLossWeighter] Weights frozen at step {self.step_count}")
        print(f"  EMA losses:")
        print(f"    CE:    {ema_ce:.4f}")
        print(f"    Recon: {ema_recon:.4f}")
        print(f"  Computed weights:")
        print(f"    CE:    {w_ce:.4f}")
        print(f"    Recon: 1.0000")
        print(f"  Expected contribution ratio:")
        print(f"    CE:    {weighted_ce/total_weighted*100:.1f}% (target: {target_ce*100:.0f}%)")
        print(f"    Recon: {weighted_recon/total_weighted*100:.1f}% (target: {target_recon*100:.0f}%)")
        print(f"{'='*60}\n")
    
    def get_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        if self.weights_frozen:
            return self.frozen_weights
        else:
            return self.warmup_weights.copy()
    
    def is_frozen(self) -> bool:
        """权重是否已冻结"""
        return self.weights_frozen
    
    def get_stats(self) -> Dict[str, any]:
        """获取统计信息"""
        return {
            'step_count': self.step_count,
            'weights_frozen': self.weights_frozen,
            'ema_losses': self.ema_losses.copy(),
            'current_weights': self.get_weights()
        }
