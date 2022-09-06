from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import lightning_getattr
import math
import torch
from torch.utils.checkpoint import checkpoint   # Without this, I get an import error.
from torch import nn
from torch.nn import functional as F
import transformers
from transformers.optimization import get_linear_schedule_with_warmup

from transformers import LongformerModel

from abstract.eval.multivers.longchecker.allennlp_nn_util import batched_index_select
from abstract.eval.multivers.longchecker.allennlp_feedforward import FeedForward
from abstract.eval.multivers.longchecker.metrics import SciFactMetrics
from abstract.eval.multivers.longchecker.util import *


def masked_binary_cross_entropy_with_logits(input, target, weight, rationale_mask):
    """
    Binary cross entropy loss. Ignore values where the target is -1. Compute
    loss as a "mean of means", first taking the mean over the sentences in each
    row, and then over all the rows.
    """
    # Mask to indicate which values contribute to loss.
    mask = torch.where(target > -1, 1, 0)

    # Need to convert target to float, and set -1 values to 0 in order for the
    # computation to make sense. We'll ignore the -1 values later.
    float_target = target.clone().to(torch.float)
    float_target[float_target == -1] = 0
    losses = F.binary_cross_entropy_with_logits(
        input, float_target, reduction="none")
    # Mask out the values that don't matter.
    losses = losses * mask

    # Take "sum of means" over the sentence-level losses for each instance.
    # Take means so that long documents don't dominate.
    # Multiply by `rationale_mask` to ignore sentences where we don't have
    # rationale annotations.
    n_sents = mask.sum(dim=1)
    totals = losses.sum(dim=1)
    means = totals / n_sents
    final_loss = (means * weight * rationale_mask).sum()

    return final_loss


class LongCheckerModel(pl.LightningModule):
    """
    Multi-task SciFact model that encodes claim / abstract pairs using
    Longformer and then predicts rationales and labels in a multi-task fashion.
    """
    def __init__(self, hparams):
        """
        Arguments are set by `add_model_specific_args`.
        """
        super().__init__()
        self.save_hyperparameters()

        # Constants
        self.nei_label = 1  # Int category for NEI label.

        # Classificaiton thresholds. These were added later, so older configs
        # won't have them.
        if hasattr(hparams, "label_threshold"):
            self.label_threshold = hparams.label_threshold
        else:
            self.label_threshold = None

        if hasattr(hparams, "rationale_threshold"):
            self.rationale_threshold = hparams.rationale_threshold
        else:
            self.rationale_threshold = 0.5

        # Paramters
        self.label_weight = hparams.label_weight
        self.rationale_weight = hparams.rationale_weight
        self.frac_warmup = hparams.frac_warmup

        # Model components.
        self.encoder_name = hparams.encoder_name
        self.encoder = self._get_encoder(hparams)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)

        # Final output layers.
        hidden_size = self.encoder.config.hidden_size
        activations = [nn.GELU(), nn.Identity()]
        dropouts = [self.dropout.p, 0]
        self.label_classifier = FeedForward(
            input_dim=hidden_size,
            num_layers=2,
            hidden_dims=[hidden_size, hparams.num_labels],
            activations=activations,
            dropout=dropouts)
        self.rationale_classifier = FeedForward(
            input_dim=2 * hidden_size,
            num_layers=2,
            hidden_dims=[hidden_size, 1],
            activations=activations,
            dropout=dropouts)

        # Learning rates.
        self.lr = hparams.lr

        # Metrics
        fold_names = ["train", "valid", "test"]
        metrics = {}
        for name in fold_names:
            metrics[f"metrics_{name}"] = SciFactMetrics(compute_on_step=False)

        self.metrics = nn.ModuleDict(metrics)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        encoder: The transformer encoder that gets the embeddings.
        label_weight: The weight to assign to label prediction in the loss function.
        rationale_weight: The weight to assign to rationale selection in the loss function.
        num_labels: The number of label categories.
        gradient_checkpointing: Whether to use gradient checkpointing with Longformer.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--encoder_name", type=str, default="allenai/longformer-base-4096")
        parser.add_argument("--label_weight", type=float, default=1.0)
        parser.add_argument("--rationale_weight", type=float, default=15.0)
        parser.add_argument("--num_labels", type=int, default=3)
        parser.add_argument("--gradient_checkpointing", action="store_true")
        parser.add_argument("--lr", type=float, default=5e-5)
        parser.add_argument("--frac_warmup", type=float, default=0.1,
                            help="The fraction of training to use for warmup.")
        parser.add_argument("--scheduler_total_epochs", default=None, type=int,
                            help="If given, pass as total # epochs to LR scheduler.")
        parser.add_argument("--label_threshold", default=None, type=float,
                            help="Threshold for non-NEI label.")
        parser.add_argument("--rationale_threshold", default=0.5, type=float,
                            help="Threshold for rationale.")

        return parser

    @staticmethod
    def _get_encoder(hparams):
        "Start from the Longformer science checkpoint."
        starting_encoder_name = "allenai/longformer-large-4096"
        encoder = LongformerModel.from_pretrained(
            starting_encoder_name,
            gradient_checkpointing=hparams.gradient_checkpointing)

        fn = os.path.expanduser('~/data_tmp/longformer_large_science.ckpt')
        checkpoint_prefixed = torch.load(fn)

        # New checkpoint
        new_state_dict = {}
        # Add items from loaded checkpoint.
        for k, v in checkpoint_prefixed.items():
            # Don't need the language model head.
            if "lm_head." in k:
                continue
            # Get rid of the first 8 characters, which say `roberta.`.
            new_key = k[8:]
            new_state_dict[new_key] = v

        # Add items from Huggingface state_dict. These are never used, but
        # they're needed to make things line up
        # ADD_TO_CHECKPOINT = ["embeddings.position_ids"]
        # for name in ADD_TO_CHECKPOINT:
        #     new_state_dict[name] = orig_state_dict[name]

        # Resize embeddings and load state dict.
        target_embed_size = new_state_dict['embeddings.word_embeddings.weight'].shape[0]
        encoder.resize_token_embeddings(target_embed_size)
        encoder.load_state_dict(new_state_dict)

        return encoder

    def forward(self, tokenized):
        """
        Run the forward pass. Encode the inputs and return softmax values for
        the labels and the rationale sentences.
        """
        # Encode.
        encoded = self.encoder(**tokenized)

        # Make label predictions.
        pooled_output = self.dropout(encoded.pooler_output)
        # [n_documents x n_labels]
        label_logits = self.label_classifier(pooled_output)

        return label_logits

    def training_step(self, batch, batch_idx):
        "Multi-task loss on a batch of inputs."
        res = self(batch["tokenized"], batch["abstract_sent_idx"])

        # Loss for label prediction.
        label_loss = F.cross_entropy(
            res["label_logits"], batch["label"], reduction="none")
        # Take weighted average of per-sample losses.
        label_loss = (batch["weight"] * label_loss).sum()

        # Loss for rationale selection.
        rationale_loss = masked_binary_cross_entropy_with_logits(
            res["rationale_logits"], batch["rationale"], batch["weight"],
            batch["rationale_mask"])

        # Loss is a weighted sum of the two components.
        loss = self.label_weight * label_loss + self.rationale_weight * rationale_loss

        # Invoke metrics.
        self.log("label_loss", label_loss)
        self.log("rationale_loss", rationale_loss)
        self.log("loss", loss)

        self._invoke_metrics(res, batch, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch["tokenized"], batch["abstract_sent_idx"])
        self._invoke_metrics(pred, batch, "valid")

    def validation_epoch_end(self, outs):
        "Log metrics at end of validation."
        # Log the train metrics here so that we keep track of train and valid
        # metrics at the same time, even if we validate multiple times an epoch.
        self._log_metrics("train")
        self._log_metrics("valid")

    def test_step(self, batch, batch_idx):
        pred = self(batch["tokenized"], batch["abstract_sent_idx"])
        self._invoke_metrics(pred, batch, "test")

    def test_epoch_end(self, outs):
        "Log metrics at end of test."
        # As above, log the train metrics together with the test metrics.
        self._log_metrics("train")
        self._log_metrics("test")

    def _invoke_metrics(self, pred, batch, fold):
        """
        Invoke metrics for a single step of train / validation / test.
        `batch` is gold, `pred` is prediction, `fold` specifies the fold.
        """
        assert fold in ["train", "valid", "test"]

        # We won't need gradients.
        detached = {k: v.detach() for k, v in pred.items()}
        # Invoke the metrics appropriate for this fold.
        self.metrics[f"metrics_{fold}"](detached, batch)

    def _log_metrics(self, fold):
        "Log metrics for this epoch."
        the_metric = self.metrics[f"metrics_{fold}"]
        to_log = the_metric.compute()
        the_metric.reset()
        for k, v in to_log.items():
            self.log(f"{fold}_{k}", v)

        # Uncomment this if still hanging.
            # self.log(f"{fold}_{k}", v, sync_dist=True, sync_dist_op="sum")

    def configure_optimizers(self):
        "Set the same LR for all parameters."
        hparams = self.hparams.hparams
        optimizer = transformers.AdamW(self.parameters(), lr=self.lr)

        # If we're debugging, just use the vanilla optimizer.
        if hparams.fast_dev_run or hparams.debug:
            return optimizer

        # Calculate total number of training steps, for the optimizer.
        steps_per_epoch = math.ceil(
            hparams.num_training_instances /
            (hparams.gpus * hparams.train_batch_size * hparams.accumulate_grad_batches))

        if hparams.scheduler_total_epochs is not None:
            n_epochs = hparams.scheduler_total_epochs
        else:
            n_epochs = hparams.max_epochs

        num_steps = n_epochs * steps_per_epoch
        warmup_steps = num_steps * self.frac_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps)

        lr_dict = {"scheduler": scheduler,
                   "interval": "step"}
        res = {"optimizer": optimizer,
               "lr_scheduler": lr_dict}

        return res

    def predict(self, batch, force_rationale=False):
        """
        Make predictions on a batch passed in from the dataloader.
        """
        # Run forward pass.
        with torch.no_grad():
            output = self(batch["tokenized"], batch["abstract_sent_idx"])

        return self.decode(output, batch, force_rationale)

    @staticmethod
    def decode(output, batch, force_rationale=False):
        """
        Run decoding to get output in human-readable form. The `output` here is
        the output of the forward pass.
        """
        # Mapping from ints to labels.
        label_lookup = {0: "CONTRADICT",
                        1: "NEI",
                        2: "SUPPORT"}

        # Get predicted rationales, only keeping eligible sentences.
        instances = unbatch(batch, ignore=["tokenized"])
        output_unbatched = unbatch(output)

        predictions = []
        for this_instance, this_output in zip(instances, output_unbatched):
            predicted_label = label_lookup[this_output["predicted_labels"]]

            # Due to minibatching, may need to get rid of padding sentences.
            rationale_ix = this_instance["abstract_sent_idx"] > 0
            rationale_indicators = this_output["predicted_rationales"][rationale_ix]
            predicted_rationale = rationale_indicators.nonzero()[0].tolist()
            # Need to convert from numpy data type to native python.
            predicted_rationale = [int(x) for x in predicted_rationale]

            # If we're forcing a rationale, then if the predicted label is not "NEI"
            # take the highest-scoring sentence as a rationale.
            if predicted_label != "NEI" and not predicted_rationale and force_rationale:
                candidates = this_output["rationale_probs"][rationale_ix]
                predicted_rationale = [candidates.argmax()]

            res = {
                "claim_id": int(this_instance["claim_id"]),
                "abstract_id": int(this_instance["abstract_id"]),
                "predicted_label": predicted_label,
                "predicted_rationale": predicted_rationale,
                "label_probs": this_output["label_probs"],
                "rationale_probs": this_output["rationale_probs"][rationale_ix]
            }
            predictions.append(res)

        return predictions


if __name__ == '__main__':
    model = LongCheckerModel.load_from_checkpoint('/home/t-gadams/multivers/script/checkpoints/scifact.ckpt', strict=False)
