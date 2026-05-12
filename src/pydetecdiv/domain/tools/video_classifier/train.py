"""
Video classifier trainer class
"""
import os
from datetime import datetime
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from pydetecdiv.app.tools.deep_learning import ModelTrainer, set_optimizer, set_schedulers
from pydetecdiv.domain.tools.video_classifier.models.MViT import MViT_v2_s, MViT_v1_b
from pydetecdiv.torch import ClassifierTrainingStats
from pydetecdiv.torch.loss import FocalLoss
from pydetecdiv.torch.metrics import set_metrics

if TYPE_CHECKING:
    from pydetecdiv.domain.tools.video_classifier import VideoClassifier


class VideoClassifierTrainer(ModelTrainer):
    """
    Video classifier trainer class to run the training of deep learning video classifier model
    """

    def __init__(self, tool: 'VideoClassifier'):
        super().__init__(tool)

    def train_model(self):
        """
        Train the video classifier model, running the training loop once per epoch for as many epochs as requested by the user
        """
        print("Training video classifier model...")
        training_dataset, validation_dataset, class_weights = self.tool.prepare_data_for_training()

        print(f'Training dataset size: {len(training_dataset)}')
        print(f'Validation dataset size: {len(validation_dataset)}')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'running training on {"GPU" if device.type == "cuda" else "CPU"}')

        torch.random.manual_seed(self.tool.parameters.seed.value)

        model = MViT_v2_s(n_classes=len(training_dataset.class_names), dropout=self.tool.parameters.dropout.value)
        model_name = 'MViT_v2_small'

        model = model.to(device)

        optimizer = set_optimizer(self.tool.parameters, model.parameters())

        train_stats = ClassifierTrainingStats(model_name=model_name, class_names=training_dataset.class_names)
        train_stats.add_metrics(set_metrics(train_stats.num_classes))
        train_stats.metrics.to(device)
        train_stats.val_metrics.to(device)
        train_stats.history.main_metric = 'MCC'
        main_metric = train_stats.history.main_metric

        training_dataloader = DataLoader(training_dataset, batch_size=self.tool.parameters.batch_size.value, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=self.tool.parameters.batch_size.value, shuffle=True)

        print(class_weights)
        print(training_dataset.class_names)
        loss_fn = FocalLoss(alpha=class_weights, gamma=1.0, reduction='mean')

        main_scheduler, reduce_on_plateau = set_schedulers(parameters=self.tool.parameters, optimizer=optimizer)

        # summary(model, (self.tool.parameters['batch_size'].value, 15, 3, 224, 224), device=device)

        run = self.tool.save_run(command='train_model')
        print(run)

        for epoch in range(self.tool.parameters['epochs'].value):
            self.training_loop(training_dataloader, validation_dataloader, model, loss_fn, optimizer, device, train_stats)
            print(f"Epoch {epoch + 1}/{self.tool.parameters['epochs'].value}, "
                  f"Training Loss: {train_stats.history.loss[-1]:.4f}, "
                  f"Validation Loss: {train_stats.history.val_loss[-1]:.4f}, "
                  f"{main_metric}: {train_stats.history.metric_history(main_metric)[-1]:.3f}, "
                  f"Val {main_metric}: {train_stats.history.val_metric_history(main_metric)[-1]:.3f}, "
                  f"learning rate: {main_scheduler.get_last_lr()[0]:0.2e}, "
                  f" -- ({datetime.now().strftime('%H:%M:%S')})")

            if train_stats.is_best_val_loss(epoch):
                checkpoint_filepath = os.path.join(self.tool.checkpoints_path(run), f'epoch{epoch}_best_loss.pt')
                model_scripted = torch.jit.script(model)
                model_scripted.save(checkpoint_filepath)
                print(f"Saving best model at epoch {epoch + 1} with val loss {train_stats.history.val_loss[-1]:.4f}"
                      f" and train loss {train_stats.history.loss[-1]:.4f}")

            main_scheduler.step()
            if reduce_on_plateau is not None:
                reduce_on_plateau.step(train_stats.history.val_loss[-1])

        checkpoint_filepath = os.path.join(self.tool.checkpoints_path(run), f'last_epoch{epoch}.pt')
        model_scripted = torch.jit.script(model)
        model_scripted.save(checkpoint_filepath)

        training_dataset.close()
        validation_dataset.close()

        return train_stats
