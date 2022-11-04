import os

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import RobertaModel, RobertaConfig, RobertaForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaClassificationHeadWithPooling


HF_TRANSFORMER = os.path.expanduser('~/RoBERTa-base-PM-M3-Voc-distill-align-hf')


class Discriminator(pl.LightningModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.save_hyperparameters(args)
        self.config = RobertaConfig.from_pretrained(HF_TRANSFORMER)
        self.config.num_labels = 1
        # self.model = RobertaForSequenceClassification(self.config)
        # self.model.roberta = self.model.roberta.from_pretrained(HF_TRANSFORMER)
        self.model = RobertaModel.from_pretrained(HF_TRANSFORMER)
        self.pooler = RobertaClassificationHeadWithPooling(self.config)
        self.tokenizer = tokenizer
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')

    def generate_logits(self, model_inputs):
        outputs = self.model(**model_inputs)[0]
        pool_mask = model_inputs['attention_mask'] == 0
        logits = self.pooler(outputs, attention_mask=pool_mask).squeeze(-1)
        return logits

    def training_step(self, batch, batch_idx):
        model_inputs = batch.pop('model_inputs')
        methods = batch.pop('methods')
        batch_size = len(model_inputs['input_ids']) // self.hparams.max_candidates
        logits = self.generate_logits(model_inputs).view(batch_size, -1)
        labels = torch.zeros(size=(batch_size, ), dtype=torch.long).to(self.device)
        loss = self.loss_func(logits, labels)
        correct = (torch.argmax(logits.detach(), dim=1) == 0).sum()
        accuracy = correct / batch_size
        self.log('train/loss', loss, on_epoch=False, on_step=True, prog_bar=True)
        self.log('train/accuracy', accuracy, on_epoch=False, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        model_inputs = batch.pop('model_inputs')
        methods = batch.pop('methods')
        batch_size = len(model_inputs['input_ids']) // self.hparams.max_candidates
        logits = self.generate_logits(model_inputs).view(batch_size, -1)

        labels = torch.zeros(size=(batch_size, ), dtype=torch.long).to(self.device)
        loss = self.loss_func(logits, labels)
        correct = (torch.argmax(logits.detach(), dim=1) == 0).sum()
        accuracy = correct / batch_size
        self.log(
            'validation/loss', loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size
        )
        self.log(
            'validation/accuracy', accuracy, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size
        )
        return loss

    def configure_optimizers(self):
        nps = list(self.named_parameters())

        pooler = [(n, p) for n, p in nps if 'pooler' in n and p.requires_grad]
        used_names = set([x for (x, _) in pooler])
        remaining = [(n, p) for n, p in nps if n not in used_names and p.requires_grad]
        grouped_parameters = [
            {
                'params': [p for n, p in remaining],
                'weight_decay': self.hparams.weight_decay,
                'lr': self.hparams.lr,
            },
            {
                'params': [p for n, p in pooler],
                'weight_decay': 0.0,
                'lr': 1e-3,
            },
        ]

        optimizer = torch.optim.AdamW(grouped_parameters)

        # 6% is somewhat standard for fine-tuning Transformers (can be a tunable hyper-parameter as well)
        # nonzero warmup helps mitigate risk of catastrophic forgetting from pre-training (big risk bc/ of new domain)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.max_steps)

        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [lr_scheduler]

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        return items
