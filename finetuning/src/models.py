import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, T5EncoderModel
from esm.models.esm3 import ESM3
from huggingface_hub import PyTorchModelHubMixin # <-- 1. Import Mixin

class L2Norm(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=-1, eps=self.eps)

# -------------------------------
# Model
# -------------------------------
# 2. Add PyTorchModelHubMixin to inheritance
class ModelForResidueClassification(nn.Module, PyTorchModelHubMixin): 
    def __init__(self, model_name: str, pos_weight=None, **kwargs):
        super().__init__()
        
        self.model_name = model_name

        # 3. Ensure pos_weight is handled nicely for JSON serialization (Hub configs require this)
        if isinstance(pos_weight, torch.Tensor):
            self.pos_weight = pos_weight
        elif pos_weight is not None:
            self.pos_weight = torch.tensor(pos_weight, dtype=torch.float)
        else:
            self.pos_weight = None

        # load model (encoder only for encoder-decoder models)
        if any(x in model_name for x in ["ankh", "T5", "t5"]):
            self.model = T5EncoderModel.from_pretrained(model_name)
            self.hidden_size = self.model.config.hidden_size
        elif "esm3" in model_name:
            self.model = ESM3.from_pretrained("esm3_sm_open_v1").float()
            self.hidden_size = self.model.encoder.sequence_embed.weight.shape[1]
        else:
            self.model = AutoModel.from_pretrained(model_name)
            self.hidden_size = self.model.config.hidden_size

        # linear layer
        self.scaler = nn.LayerNorm(self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # compute logits
        if "esm3" in self.model_name:
            outputs = self.model.forward(sequence_tokens=input_ids)
            hidden_states = outputs.embeddings
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
        
        # Apply the scaling
        scaled_hidden_states = self.scaler(hidden_states)

        logits = self.classifier(scaled_hidden_states).squeeze(-1)
        
        # compute loss
        loss = None
        if labels is not None:
            mask = labels != -100
            logits_masked = logits[mask]
            labels_masked = labels[mask]
            
            # Handle pos_weight device dynamically
            pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
            loss = loss_fn(logits_masked, labels_masked)

        return {"loss": loss, "logits": logits}