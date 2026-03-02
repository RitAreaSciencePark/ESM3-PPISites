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
from transformers import PretrainedConfig

class ResidueClassificationConfig(PretrainedConfig):
    model_type = "residue-classification"

    def __init__(
        self,
        model_name="facebook/esm2_t33_650M_UR50D",
        pos_weight=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.pos_weight = pos_weight
        
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel, T5EncoderModel

class ModelForResidueClassification(PreTrainedModel):
    config_class = ResidueClassificationConfig

    def __init__(self, model_name, pos_weight=None):
        config = ResidueClassificationConfig(model_name, pos_weight)
        super().__init__(config)

        self.model_name = config.model_name
        self.pos_weight = (
            torch.tensor(config.pos_weight)
            if config.pos_weight is not None
            else None
        )

        # Load backbone
        if any(x in self.model_name for x in ["ankh", "T5", "t5"]):
            self.model = T5EncoderModel.from_pretrained(self.model_name)
            self.hidden_size = self.model.config.hidden_size

        elif "esm3" in self.model_name:
            from esm.models.esm3 import ESM3
            self.model = ESM3.from_pretrained("esm3_sm_open_v1").float()
            self.hidden_size = self.model.encoder.sequence_embed.weight.shape[1]

        else:
            self.model = AutoModel.from_pretrained(self.model_name)
            self.hidden_size = self.model.config.hidden_size

        self.scaler = nn.LayerNorm(self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, 1)

        # VERY IMPORTANT
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None):

        if "esm3" in self.model_name:
            outputs = self.model(sequence_tokens=input_ids)
            hidden_states = outputs.embeddings
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = outputs.last_hidden_state

        scaled_hidden_states = self.scaler(hidden_states)
        logits = self.classifier(scaled_hidden_states).squeeze(-1)

        loss = None
        if labels is not None:
            mask = labels != -100
            logits_masked = logits[mask]
            labels_masked = labels[mask]

            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(logits.device)
                if self.pos_weight is not None
                else None
            )
            loss = loss_fn(logits_masked, labels_masked)

        return {"loss": loss, "logits": logits}