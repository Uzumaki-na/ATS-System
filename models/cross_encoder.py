"""
Model 1: Cross-Encoder for Deep Resume-Job Scoring
Multi-head architecture for regression and classification.
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Tuple, Optional
from config import config


class MultiHeadCrossEncoder(nn.Module):
    """
    Cross-Encoder for resume-job pair scoring.
    Combines BERT with multi-head output for:
    - Regression: Fit score (0-1)
    - Classification: Good/Potential/Bad fit
    """
    
    def __init__(
        self,
        pretrained_model_name: str = "bert-base-uncased",
        num_classes: int = 3,
        hidden_size: int = 768,
        dropout: float = 0.1,
        regression_head_units: int = 256,
        classification_head_units: int = 128,
        freeze_base: bool = False
    ):
        super().__init__()
        
        self.pretrained_model_name = pretrained_model_name
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # Load pretrained transformer
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        
        # Optionally freeze base model layers
        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Get encoder output size
        encoder_output_size = self.encoder.config.hidden_size
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Regression Head for scoring
        self.regression_head = nn.Sequential(
            nn.Linear(encoder_output_size, regression_head_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(regression_head_units, regression_head_units // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(regression_head_units // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        # Classification Head for labels
        self.classification_head = nn.Sequential(
            nn.Linear(encoder_output_size, classification_head_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classification_head_units, classification_head_units // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classification_head_units // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for custom heads."""
        for module in self.regression_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        for module in self.classification_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the cross-encoder.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_dict: Whether to return dict or tuple
            
        Returns:
            Dictionary with regression score and classification logits
        """
        # Get encoder outputs
        encoder_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'return_dict': return_dict
        }

        if token_type_ids is not None:
            encoder_kwargs['token_type_ids'] = token_type_ids

        try:
            encoder_outputs = self.encoder(**encoder_kwargs)
        except TypeError:
            encoder_kwargs.pop('token_type_ids', None)
            encoder_outputs = self.encoder(**encoder_kwargs)
        
        # Get [CLS] token representation (first token)
        if return_dict:
            sequence_output = encoder_outputs.last_hidden_state
        else:
            sequence_output = encoder_outputs[0]
        
        cls_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        
        # Apply dropout
        cls_output = self.dropout(cls_output)
        
        # Get predictions from both heads
        regression_score = self.regression_head(cls_output).squeeze(-1)  # [batch_size]
        classification_logits = self.classification_head(cls_output)  # [batch_size, num_classes]
        
        return {
            'regression_score': regression_score,
            'classification_logits': classification_logits,
            'cls_embedding': cls_output
        }
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get predictions from the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Dictionary with score, predicted class, and probabilities
        """
        outputs = self.forward(input_ids, attention_mask, token_type_ids=token_type_ids)
        
        # Get classification predictions
        probs = torch.softmax(outputs['classification_logits'], dim=-1)
        predicted_class = torch.argmax(probs, dim=-1)
        
        return {
            'score': outputs['regression_score'],
            'predicted_class': predicted_class,
            'probabilities': probs,
            'logits': outputs['classification_logits']
        }
    
    def get_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_score: torch.Tensor,
        target_class: torch.Tensor,
        regression_weight: float = 0.5,
        classification_weight: float = 0.5,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate combined loss for training.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            target_score: Target regression scores
            target_class: Target class labels
            regression_weight: Weight for regression loss
            classification_weight: Weight for classification loss
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        outputs = self.forward(input_ids, attention_mask, token_type_ids=token_type_ids)
        
        # Regression loss (MSE)
        regression_loss_fn = nn.MSELoss()
        regression_loss = regression_loss_fn(outputs['regression_score'], target_score)
        
        # Classification loss (Cross-Entropy)
        classification_loss_fn = nn.CrossEntropyLoss()
        classification_loss = classification_loss_fn(
            outputs['classification_logits'],
            target_class
        )
        
        # Combined loss
        total_loss = (
            regression_weight * regression_loss + 
            classification_weight * classification_loss
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'regression_loss': regression_loss.item(),
            'classification_loss': classification_loss.item()
        }
        
        return total_loss, loss_dict
    
    @classmethod
    def from_config(cls, model_config: Dict = None) -> 'MultiHeadCrossEncoder':
        """Create model from configuration."""
        if model_config is None:
            model_config = config['models']['cross_encoder']
        
        return cls(
            pretrained_model_name=model_config['pretrained_model'],
            num_classes=model_config.get('num_classes', 3),
            hidden_size=model_config.get('hidden_size', 768),
            dropout=model_config.get('dropout', 0.1),
            regression_head_units=model_config.get('regression_head_units', 256),
            classification_head_units=model_config.get('classification_head_units', 128)
        )
    
    def save(self, path: str):
        """Save model weights and config."""
        tmp_path = f"{path}.tmp"
        payload = {
            'model_state_dict': self.state_dict(),
            'pretrained_model_name': self.pretrained_model_name,
            'num_classes': self.num_classes,
            'hidden_size': self.hidden_size
        }

        try:
            torch.save(payload, tmp_path, _use_new_zipfile_serialization=False)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    
    @classmethod
    def load(cls, path: str, device: str = 'cuda') -> 'MultiHeadCrossEncoder':
        """Load model weights."""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            pretrained_model_name=checkpoint['pretrained_model_name'],
            num_classes=checkpoint['num_classes'],
            hidden_size=checkpoint['hidden_size']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model


class CrossEncoderTrainer:
    """Training utilities for Cross-Encoder."""
    
    def __init__(
        self,
        model: MultiHeadCrossEncoder,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        gradient_accumulation_steps: int = 4,
        fp16: bool = False,
        device: str = 'cuda'
    ):
        self.model = model
        self.device = device
        self.fp16 = fp16
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Optimizer setup
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        if warmup_steps and warmup_steps > 0:
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0,
                total_iters=1
            )
        
        # Gradient scaler for FP16
        self.scaler = torch.cuda.amp.GradScaler() if fp16 else None
    
    def train_epoch(
        self,
        dataloader,
        regression_weight: float = 0.5,
        classification_weight: float = 0.5
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_regression_loss = 0
        total_classification_loss = 0
        loss_counts = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            target_score = batch['score'].to(self.device)
            target_class = batch['label'].to(self.device)
            
            # Mixed precision training
            if self.fp16:
                with torch.cuda.amp.autocast():
                    loss, loss_dict = self.model.get_loss(
                        input_ids, attention_mask,
                        target_score, target_class,
                        regression_weight, classification_weight,
                        token_type_ids=token_type_ids
                    )
                    loss = loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            else:
                loss, loss_dict = self.model.get_loss(
                    input_ids, attention_mask,
                    target_score, target_class,
                    regression_weight, classification_weight,
                    token_type_ids=token_type_ids
                )
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            
            total_loss += loss_dict['total_loss']
            total_regression_loss += loss_dict.get('regression_loss', 0.0)
            total_classification_loss += loss_dict.get('classification_loss', 0.0)
            loss_counts += 1
        
        return {
            'avg_loss': total_loss / loss_counts,
            'regression_loss': total_regression_loss / loss_counts,
            'classification_loss': total_classification_loss / loss_counts
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_scores = []
        all_score_targets = []
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            target_score = batch['score'].to(self.device)
            target_class = batch['label'].to(self.device)
            
            predictions = self.model.predict(input_ids, attention_mask, token_type_ids=token_type_ids)
            
            # Calculate losses
            _, loss_dict = self.model.get_loss(
                input_ids, attention_mask,
                target_score, target_class,
                token_type_ids=token_type_ids
            )
            
            total_loss += loss_dict['total_loss']
            all_predictions.extend(predictions['predicted_class'].cpu().tolist())
            all_targets.extend(target_class.cpu().tolist())
            all_scores.extend(predictions['score'].cpu().tolist())
            all_score_targets.extend(target_score.cpu().tolist())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
        
        metrics = {
            'val_loss': total_loss / len(dataloader),
            'accuracy': accuracy_score(all_targets, all_predictions),
            'mse': mean_squared_error(all_score_targets, all_scores),
            'f1_macro': f1_score(all_targets, all_predictions, average='macro'),
            'f1_weighted': f1_score(all_targets, all_predictions, average='weighted')
        }
        
        return metrics
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save training checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }, path)
