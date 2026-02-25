import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel , T5EncoderModel
from esm.models.esm3 import ESM3



class L2Norm(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # Normalize along the last dimension (hidden_size)
        return F.normalize(x, p=2, dim=-1, eps=self.eps)

# -------------------------------
# Model
# -------------------------------
class ModelForResidueClassification(nn.Module):
    def __init__(self, model_name, pos_weight=None):
        super().__init__()
        
        self.pos_weight = pos_weight
        self.model_name = model_name

        #load model (encoder only for encoder-decoder models)
        if any(x in model_name for x in ["ankh", "T5", "t5"]):
            self.model = T5EncoderModel.from_pretrained(model_name)
            self.hidden_size = self.model.config.hidden_size
        elif "esm3" in model_name:
            self.model = ESM3.from_pretrained("esm3_sm_open_v1").float()
            self.hidden_size = self.model.encoder.sequence_embed.weight.shape[1]
        else:
            self.model = AutoModel.from_pretrained(model_name)
            self.hidden_size = self.model.config.hidden_size

        #linear layer
        self.scaler = nn.LayerNorm(self.hidden_size)
        #self.scaler = L2Norm()
        self.classifier = nn.Linear(self.hidden_size, 1)
        
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        #compute logits
        if "esm3" in self.model_name:
            outputs = self.model.forward(sequence_tokens = input_ids)
            hidden_states = outputs.embeddings
        else:
            outputs = self.model(input_ids=input_ids, 
                                 attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
        
        # --- Apply Standard Scaling ---
        # hidden_states shape is (batch_size, seq_len, hidden_size)
        #B, S, H = hidden_states.shape
        
        # BatchNorm1d expects input as (N, C) or (N, C, L).
        # We'll treat the batch and sequence dims as one 'N' dimension
        # and hidden_size as the 'C' (channels/features) dimension.
        #hidden_states_reshaped = hidden_states.reshape(B * S, H)
        
        # Apply the scaling
        scaled_hidden_states = self.scaler(hidden_states)
        
        # Reshape back to the original (B, S, H)
        #scaled_hidden_states = scaled_hidden_states.reshape(B, S, H)
        # --- End Scaling ---



        #logits = self.classifier(hidden_states).squeeze(-1)
        logits = self.classifier(scaled_hidden_states).squeeze(-1)
        #compute loss
        loss = None
        if labels is not None:
            mask = labels != -100
            logits_masked = logits[mask]
            labels_masked = labels[mask]
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
            loss = loss_fn(logits_masked, labels_masked)

        #return
        return {"loss": loss, "logits": logits}