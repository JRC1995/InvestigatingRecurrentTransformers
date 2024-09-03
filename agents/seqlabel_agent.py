import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
from controllers import *


class seqlabel_agent:
    def __init__(self, model, config, device):
        self.model = model
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = eval(config["optimizer"])
        self.global_step = 0

        grouped_parameters = [{'params': [p for n, p in model.named_parameters() if p.requires_grad],
                               'weight_decay': config["weight_decay"], 'lr': config["lr"]}]

        self.optimizer = optimizer(grouped_parameters,
                                   lr=config["lr"],
                                   betas=config["betas"],
                                   weight_decay=config["weight_decay"],
                                   eps=config["eps"])

        self.amp_scaler = torch.cuda.amp.GradScaler(enabled=True)

        self.scheduler, self.epoch_scheduler = get_scheduler(config, self.optimizer)
        #self.scheduler = None
        #self.epoch_level_scheduler = False
        self.config = config
        self.device = device
        self.DataParallel = config["DataParallel"]
        self.optimizer.zero_grad()

        if self.config["classes_num"] > 2:
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    # %%
    def loss_fn(self, logits, labels, loss_masks, train=False, aux_loss=None):

        if self.config["classes_num"] == 2:
            labels = F.one_hot(labels, num_classes=2).float()
            labels = labels[..., 1].unsqueeze(-1)

            loss = self.criterion(logits, labels)

            loss = loss.squeeze(-1)

        else:
            N, S, C = logits.size()
            logits = logits.view(N * S, C)
            labels = labels.view(N * S)
            loss = self.criterion(logits, labels)
            #print("loss size: ", loss.size())
            #assert loss.size() == (N * S)
            loss = loss.view(N, S)

        assert loss.size() == loss_masks.size()
        loss = T.sum(loss_masks * loss) / T.sum(loss_masks)
        if aux_loss is not None and train:
            loss = loss + aux_loss
        return loss

    # %%
    def run(self, batch, train=True):

        if train:
            self.model = self.model.train()
        else:
            self.model = self.model.eval()

        if not self.DataParallel:
            batch["sequences_vec"] = batch["sequences_vec"].to(self.device)
            batch["labels"] = batch["labels"].to(self.device)
            batch["input_masks"] = batch["input_masks"].to(self.device)
            batch["loss_masks"] = batch["loss_masks"].to(self.device)

        N = batch["sequences_vec"].size(0)

        with T.amp.autocast(device_type='cuda', dtype=T.float16):
            output_dict = self.model(batch)
            logits = output_dict["logits"]
            labels = batch["labels"].to(logits.device)
            loss_masks = batch["loss_masks"].to(logits.device)
            aux_loss = output_dict["aux_loss"]
            #print("labels: ", labels.size())
            #print("logits: ", logits.size())
            loss = self.loss_fn(logits=logits, labels=labels,
                                loss_masks=loss_masks,
                                train=train, aux_loss=aux_loss)

        if self.config["classes_num"] == 2:
            predictions = T.where(T.sigmoid(logits) >= 0.5,
                                  T.ones_like(logits).int().to(logits.device),
                                  T.zeros_like(logits).int().to(logits.device))
            predictions = predictions.squeeze(-1)
        else:
            predictions = T.argmax(logits, dim=-1)
        predictions = predictions.detach().cpu().numpy().tolist()

        labels = batch["labels"].cpu().numpy().tolist()
        loss_masks = batch["loss_masks"].cpu().numpy().tolist()
        metrics = self.eval_fn(predictions, labels, loss_masks)
        metrics["loss"] = loss.item()

        items = {"display_items": {"sequences": batch["sequences"],
                                   "predictions": predictions,
                                   "loss_masks": loss_masks,
                                   "labels": labels},
                 "loss": loss,
                 "metrics": metrics}

        return items

    # %%
    def backward(self, loss):
        self.amp_scaler.scale(loss).backward()
        #loss.backward()

    # %%
    def step(self):
        if self.config["max_grad_norm"] is not None:
            self.amp_scaler.unscale_(self.optimizer)
            T.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
        self.amp_scaler.step(self.optimizer)
        self.amp_scaler.update()
        #self.optimizer.step()
        self.optimizer.zero_grad()

        if self.scheduler is not None:
            self.scheduler.step()

        self.config["current_lr"] = self.optimizer.param_groups[-1]["lr"]


    # %%
    def eval_fn(self, predictions, labels, label_masks):
        total = 0
        last_total = 0
        correct = 0
        last_correct = 0
        for prediction, label, label_mask in zip(predictions, labels, label_masks):
            last_total += 1
            last_correct_state = 0
            for p, l, m in zip(prediction, label, label_mask):
                if m == 1:
                    total += 1
                    if p == l:
                        #print("p: ", p)
                        #print("l: ", l)
                        correct += 1
                        last_correct_state = 1
                    else:
                        last_correct_state = 0
            last_correct += last_correct_state

        accuracy = correct / total
        accuracy *= 100

        last_accuracy = last_correct / last_total
        last_accuracy *= 100

        return {"correct_predictions": correct,
                "total": total,
                "last_correct_predictions": last_correct,
                "last_total": last_total,
                "accuracy": accuracy,
                "last_accuracy": last_accuracy}
