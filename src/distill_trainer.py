"""
知识蒸馏分割训练器
将教师模型的知识蒸馏到学生模型
"""
import sys
from pathlib import Path

# 确保项目根目录在路径中
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.utils import LOGGER, RANK, DEFAULT_CFG
from ultralytics.utils.torch_utils import autocast


class DistillSegmentationTrainer(SegmentationTrainer):
    """知识蒸馏分割训练器
    
    支持特征蒸馏和输出蒸馏，将教师模型的知识转移到学生模型。
    
    Args:
        teacher_weights: 教师模型权重路径
        alpha: 蒸馏损失权重 (0-1)，越大蒸馏权重越高
        temperature: 软标签温度，越高分布越平滑
    """
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None,
                 teacher_weights=None, alpha=0.5, temperature=3.0):
        super().__init__(cfg, overrides, _callbacks)
        
        self.teacher_weights = teacher_weights
        self.alpha = alpha
        self.temperature = temperature
        self.teacher_model = None
        
        LOGGER.info(f"知识蒸馏训练器初始化:")
        LOGGER.info(f"  教师权重: {teacher_weights}")
        LOGGER.info(f"  蒸馏权重 alpha: {alpha}")
        LOGGER.info(f"  温度 T: {temperature}")
    
    def _setup_train(self):
        """设置训练，加载教师模型"""
        super()._setup_train()
        
        # 加载教师模型
        if self.teacher_weights:
            from ultralytics import YOLO
            LOGGER.info(f"加载教师模型: {self.teacher_weights}")
            
            teacher = YOLO(self.teacher_weights)
            self.teacher_model = teacher.model.to(self.device)
            self.teacher_model.eval()
            
            # 冻结教师模型
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            
            LOGGER.info(f"教师模型加载完成，参数已冻结")
    
    def _do_train(self):
        """训练循环，加入蒸馏损失"""
        # 调用父类训练设置
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()
        
        import warnings
        import time
        import numpy as np
        from torch import distributed as dist
        from ultralytics.utils import TQDM, colorstr
        from ultralytics.utils.torch_utils import unwrap_model
        
        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting distillation training for {self.epochs} epochs..."
        )
        
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        
        epoch = self.start_epoch
        self.optimizer.zero_grad()
        
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()
            
            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            
            pbar = enumerate(self.train_loader)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()
            
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                ni = i + nb * epoch
                
                # Warmup
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                
                # Forward with distillation
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    
                    # 学生模型前向
                    if self.args.compile:
                        preds = self.model(batch["img"])
                        hard_loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                    else:
                        hard_loss, self.loss_items = self.model(batch)
                    
                    # 蒸馏损失
                    if self.teacher_model is not None:
                        distill_loss = self._compute_distill_loss(batch)
                        # 组合损失: (1-alpha)*hard_loss + alpha*distill_loss
                        self.loss = (1 - self.alpha) * hard_loss.sum() + self.alpha * distill_loss
                        
                        # 调试输出（每个epoch第一个batch）
                        if i == 0:
                            LOGGER.info(f"[Distill Debug] Epoch {epoch+1}: hard_loss={hard_loss.sum().item():.4f}, distill_loss={distill_loss.item() if hasattr(distill_loss, 'item') else distill_loss:.4f}, total_loss={self.loss.item():.4f}")
                    else:
                        self.loss = hard_loss.sum()
                    
                    if RANK != -1:
                        self.loss *= self.world_size
                    self.tloss = self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)
                
                # Backward
                self.scaler.scale(self.loss).backward()
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni
                    
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:
                            break
                
                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                            batch["cls"].shape[0],
                            batch["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)
                
                self.run_callbacks("on_train_batch_end")
            
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")
            
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
            
            # Validation
            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5)
                self.metrics, self.fitness = self.validate()
            
            # NaN recovery
            if self._handle_nan_recovery(epoch):
                continue
            
            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)
                
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")
            
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)
            
            if RANK != -1:
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            if self.stop:
                break
            epoch += 1
        
        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        from ultralytics.utils.torch_utils import unset_deterministic
        unset_deterministic()
        self.run_callbacks("teardown")
    
    def _compute_distill_loss(self, batch):
        """计算蒸馏损失
        
        使用CWD(Channel-wise Knowledge Distillation)方法
        论文: Channel-wise Knowledge Distillation for Dense Prediction
        核心思想: 对每个通道的空间分布进行KL散度对齐
        """
        with torch.no_grad():
            # 教师模型预测（不需要梯度）
            teacher_preds = self.teacher_model(batch["img"])
        
        # 学生模型预测
        student_preds = self.model(batch["img"]) if not self.args.compile else self.model(batch["img"])
        
        # 计算CWD蒸馏损失
        distill_loss = 0.0
        T = self.temperature  # 温度参数
        
        if isinstance(student_preds, (list, tuple)) and isinstance(teacher_preds, (list, tuple)):
            for s_pred, t_pred in zip(student_preds, teacher_preds):
                if isinstance(s_pred, torch.Tensor) and isinstance(t_pred, torch.Tensor):
                    # 使用CWD计算损失
                    distill_loss += self._cwd_loss(s_pred, t_pred, T)
        elif isinstance(student_preds, torch.Tensor) and isinstance(teacher_preds, torch.Tensor):
            distill_loss = self._cwd_loss(student_preds, teacher_preds, T)
        
        return distill_loss
    
    def _cwd_loss(self, student_feat, teacher_feat, T=4.0):
        """Channel-wise Knowledge Distillation Loss
        
        对每个通道计算空间分布的KL散度
        
        Args:
            student_feat: 学生特征 [B, C_s, H, W]
            teacher_feat: 教师特征 [B, C_t, H, W]
            T: 温度参数
        """
        B = student_feat.shape[0]
        
        # 如果通道数不同，需要对齐
        if student_feat.shape[1] != teacher_feat.shape[1]:
            # 使用自适应平均池化对齐空间维度，然后用全局池化
            s_feat = F.adaptive_avg_pool2d(student_feat, 1).view(B, -1)
            t_feat = F.adaptive_avg_pool2d(teacher_feat, 1).view(B, -1)
            # 对齐到相同维度
            min_c = min(s_feat.shape[1], t_feat.shape[1])
            s_feat = s_feat[:, :min_c]
            t_feat = t_feat[:, :min_c]
            
            # 计算KL散度
            s_dist = F.log_softmax(s_feat / T, dim=1)
            t_dist = F.softmax(t_feat / T, dim=1)
            return F.kl_div(s_dist, t_dist, reduction='batchmean') * (T ** 2)
        
        # 空间维度对齐
        if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
            teacher_feat = F.interpolate(
                teacher_feat, 
                size=student_feat.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        C = student_feat.shape[1]
        
        # 展平空间维度: [B, C, H*W]
        s_flat = student_feat.view(B, C, -1)
        t_flat = teacher_feat.view(B, C, -1)
        
        # 对每个通道计算空间分布的softmax
        s_dist = F.log_softmax(s_flat / T, dim=2)
        t_dist = F.softmax(t_flat / T, dim=2)
        
        # 计算KL散度
        loss = F.kl_div(s_dist, t_dist, reduction='batchmean') * (T ** 2)
        
        return loss
