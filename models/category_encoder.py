"""
Model 2: Category Encoder for Job Category Classification
Lightweight transformer for semantic category validation.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel
from typing import Dict, List, Optional
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from config import config


class CategoryEncoder(nn.Module):
    """
    Category Encoder for classifying resumes into job categories.
    Uses a lightweight transformer (DistilBERT) for efficiency.
    """
    
    def __init__(
        self,
        pretrained_model_name: str = "distilbert-base-uncased",
        num_classes: int = 24,
        hidden_size: int = 768,
        dropout: float = 0.1,
        freeze_base: bool = False
    ):
        super().__init__()
        
        self.pretrained_model_name = pretrained_model_name
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # Load pretrained transformer
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        
        # Optionally freeze base model
        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Get encoder output size
        encoder_output_size = self.encoder.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(encoder_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the category encoder.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_dict: Whether to return dict or tuple
            
        Returns:
            Dictionary with logits and probabilities
        """
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict
        )
        
        # Get [CLS] token representation
        if return_dict:
            sequence_output = encoder_outputs.last_hidden_state
        else:
            sequence_output = encoder_outputs[0]
        
        cls_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        
        # Apply dropout and classification
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        return {
            'logits': logits,
            'cls_embedding': cls_output
        }
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get category predictions.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Dictionary with predicted class and probabilities
        """
        outputs = self.forward(input_ids, attention_mask)
        
        probs = torch.softmax(outputs['logits'], dim=-1)
        predicted_class = torch.argmax(probs, dim=-1)
        
        # Get confidence score
        confidence = torch.max(probs, dim=-1)[0]
        
        return {
            'predicted_class': predicted_class,
            'probabilities': probs,
            'confidence': confidence,
            'logits': outputs['logits']
        }
    
    def set_class_weights(self, weights: torch.Tensor) -> None:
        """Store class weights for weighted cross-entropy loss.

        Args:
            weights: Float tensor of shape (num_classes,) with inverse-frequency
                     or custom per-class weights.
        """
        self.register_buffer('_class_weights', weights)

    def get_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate cross-entropy loss (optionally class-weighted).

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            target_class: Target class labels

        Returns:
            Cross-entropy loss
        """
        outputs = self.forward(input_ids, attention_mask)
        weight = getattr(self, '_class_weights', None)
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        return loss_fn(outputs['logits'], target_class)
    
    @classmethod
    def from_config(cls, model_config: Dict = None) -> 'CategoryEncoder':
        """Create model from configuration."""
        if model_config is None:
            model_config = config['models']['category_encoder']
        
        return cls(
            pretrained_model_name=model_config['pretrained_model'],
            num_classes=model_config.get('num_classes', 24),
            hidden_size=model_config.get('hidden_size', 768),
            dropout=model_config.get('dropout', 0.1)
        )
    
    def save(self, path: str, category_to_idx: Optional[Dict[str, int]] = None):
        """Save model weights and config.

        Args:
            path: destination .pt path
            category_to_idx: optional {category_name: idx} mapping used during
                training. Persisted so load() can restore the exact label space
                the model was trained with (B3: prevents id_to_name KeyError
                when a checkpoint has 25 classes but config has 24).
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'pretrained_model_name': self.pretrained_model_name,
            'num_classes': self.num_classes,
            'hidden_size': self.hidden_size,
            'category_to_idx': category_to_idx
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cuda') -> 'CategoryEncoder':
        """Load model weights."""
        checkpoint = torch.load(path, map_location=device)

        model = cls(
            pretrained_model_name=checkpoint['pretrained_model_name'],
            num_classes=checkpoint['num_classes'],
            hidden_size=checkpoint['hidden_size']
        )

        state = checkpoint['model_state_dict']
        # Filter out saved training buffers not in the model's parameter keys
        expected = set(model.state_dict().keys())
        state = {k: v for k, v in state.items() if k in expected}
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()

        # B3: restore the label mapping the model was trained with so the
        # pipeline indexes predictions against the right id_to_name instead of
        # trusting config['categories'] (which can have a different class count).
        cat2idx = checkpoint.get('category_to_idx')
        if isinstance(cat2idx, dict):
            model.category_to_idx = {str(k): int(v) for k, v in cat2idx.items()}
            model.id_to_name = {int(v): k for k, v in model.category_to_idx.items()}
        else:
            model.category_to_idx = None
            model.id_to_name = None

        return model


class CategoryEncoderTrainer:
    """Training utilities for Category Encoder."""

    def __init__(
        self,
        model: CategoryEncoder,
        learning_rate: float = 3e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 200,
        device: str = 'cuda',
        use_amp: bool = False,
        gradient_accumulation_steps: int = 1,
        total_training_steps: int | None = None,
    ):
        self.model = model
        self.device = device
        self.use_amp = use_amp and device.type == 'cuda'
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler: warmup then cosine decay to 1e-6
        if warmup_steps and warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            if total_training_steps and total_training_steps > warmup_steps:
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_training_steps - warmup_steps,
                    eta_min=1e-6,
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_steps],
                )
            else:
                self.scheduler = warmup
        else:
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=1.0, total_iters=1
            )
    
    def train_epoch(self, dataloader, epoch: int = 1) -> Dict[str, float]:
        """Train for one epoch with FP16/AMP and gradient accumulation."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        running_loss = 0.0

        # GradScaler for FP16 — only instantiated when use_amp is True
        scaler = torch.amp.GradScaler(enabled=self.use_amp)

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            unit="batch",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]  {postfix}"
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            target_class = batch['label'].to(self.device)

            # Forward pass with optional mixed precision
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                loss = self.model.get_loss(input_ids, attention_mask, target_class)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass with optional GradScaler
            if self.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            loss_val = loss.item() * self.gradient_accumulation_steps

            # Gradient accumulation: step optimizer every N batches
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

            total_loss += loss_val
            num_batches += 1
            running_loss += loss_val

            # Update progress bar
            if (batch_idx + 1) % max(1, len(dataloader) // 100) == 0:
                pbar.set_postfix(loss=f"{running_loss / max(1, len(dataloader) // 100):.4f}")
                running_loss = 0.0

        pbar.close()

        return {'avg_loss': total_loss / num_batches}
    
    @torch.no_grad()
    def evaluate(self, dataloader, id_to_name: Dict = None) -> Dict[str, float]:
        """Evaluate model on validation set (supports FP16/AMP)."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_confidences = []

        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            target_class = batch['label'].to(self.device)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                # Get predictions
                predictions = self.model.predict(input_ids, attention_mask)
                # Calculate loss
                loss = self.model.get_loss(input_ids, attention_mask, target_class)

            total_loss += loss.item()
            all_predictions.extend(predictions['predicted_class'].cpu().tolist())
            all_targets.extend(target_class.cpu().tolist())
            all_confidences.extend(predictions['confidence'].cpu().tolist())
        
        # Calculate metrics
        metrics = {
            'val_loss': total_loss / len(dataloader),
            'accuracy': accuracy_score(all_targets, all_predictions),
            'f1_macro': f1_score(all_targets, all_predictions, average='macro'),
            'f1_weighted': f1_score(all_targets, all_predictions, average='weighted'),
            'avg_confidence': np.mean(all_confidences)
        }
        
        # Add confusion matrix if id_to_name provided
        if id_to_name is not None:
            cm = confusion_matrix(all_targets, all_predictions)
            metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def predict_single(
        self,
        text: str,
        tokenizer,
        id_to_name: Dict[int, str]
    ) -> Dict:
        """Predict category for a single resume."""
        self.model.eval()
        
        # Tokenize
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model.predict(input_ids, attention_mask)
        
        pred_class = predictions['predicted_class'].item()
        confidence = predictions['confidence'].item()
        probabilities = predictions['probabilities'].cpu().numpy()[0]
        
        return {
            'predicted_category_id': pred_class,
            'predicted_category_name': id_to_name[pred_class],
            'confidence': confidence,
            'all_probabilities': {
                id_to_name[i]: float(probabilities[i]) 
                for i in range(len(probabilities))
            }
        }
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save training checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }, path)
