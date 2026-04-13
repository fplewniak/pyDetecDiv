"""
Video classifier trainer class
"""
from datetime import datetime
from typing import TYPE_CHECKING

import polars
import torch
from torch import GradScaler, autocast
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.transforms import v2

from pydetecdiv.app import pydetecdiv_project, PyDetecDiv
from pydetecdiv.app.gui.core.widgets.viewers.plots import MatplotViewer
from pydetecdiv.app.tools.deep_learning import ModelTrainer
from pydetecdiv.domain.tools.video_classifier.models.MViT import MViT_v2_s, MViT_v1_b
from pydetecdiv.torch import ClassifierTrainingStats, set_optimizer
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

        img, target = training_dataset[0]
        roi_id, frame = training_dataset.get_ref(0)
        print(f'{roi_id}: {training_dataset.indices[0]}')
        print(training_dataset.roi(training_dataset.indices[0]['roi'].item()))
        print(img.shape, img[7].shape, target)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'running training on {"GPU" if device.type == "cuda" else "CPU"}')

        torch.random.manual_seed(self.tool.parameters['seed'].value)

        model = MViT_v2_s(n_classes=6)
        model_name = 'MViT_v2_small'

        model = model.to(device)

        optimizer = set_optimizer(self.tool.parameters, model.parameters())

        train_stats = ClassifierTrainingStats(model_name=model_name, class_names=training_dataset.class_names)
        train_stats.add_metrics(set_metrics(train_stats.num_classes))
        train_stats.metrics.to(device)
        train_stats.val_metrics.to(device)
        train_stats.history.main_metric = 'MCC'
        main_metric = train_stats.history.main_metric

        training_dataloader = DataLoader(training_dataset, batch_size=self.tool.parameters['batch_size'].value, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=self.tool.parameters['batch_size'].value, shuffle=True)

        print(class_weights)
        print(training_dataset.class_names)
        loss_fn = FocalLoss(alpha=class_weights, gamma=1.0, reduction='mean')

        summary(model, (self.tool.parameters['batch_size'].value, 15, 3, 224, 224), device=device)

        for epoch in range(self.tool.parameters['epochs'].value):
            self.training_loop(training_dataloader, validation_dataloader, model, loss_fn, optimizer, device, train_stats)
            print(f"Epoch {epoch + 1}/{self.tool.parameters['epochs'].value}, "
                  f"Training Loss: {train_stats.history.loss[-1]:.4f}, "
                  f"Validation Loss: {train_stats.history.val_loss[-1]:.4f}, "
                  f"{main_metric}: {train_stats.history.metric_history(main_metric)[-1]:.3f}, "
                  f"Val {main_metric}: {train_stats.history.metric_history.val_metric_history(main_metric)[-1]:.3f}, "
                  # f"learning rate: {scheduler.get_last_lr()[0]:0.2e}, "
                  f" -- ({datetime.now().strftime('%H:%M:%S')})")


        # seqlen = img.shape[0]
        # plot_viewer = MatplotViewer(PyDetecDiv.main_window.active_subwindow, columns=seqlen, rows=1)
        # for i in range(seqlen):
        #     img_channel_last = torch.as_tensor(img[i].permute([1, 2, 0]))
        #     plot_viewer.axes[i].imshow(img_channel_last)
        #     if i == int(seqlen / 2):
        #         plot_viewer.axes[i].set_title(f'{training_dataset.class_names[target - 1]}')
        #     plot_viewer.axes[i].set_xlabel(f'{frame + i}')
        # tab = PyDetecDiv.main_window.add_tabbed_window(f'{PyDetecDiv.project_name} / {roi_id}')
        # tab.project_name = PyDetecDiv.project_name
        # tab.addTab(plot_viewer, 'Sample sequence')
        # tab.setCurrentWidget(plot_viewer)

        training_dataset.close()
        validation_dataset.close()

        #run = self.tool.save_run(command='training')

    # def training_loop(self, training_dataloader, validation_dataloader, model, loss_fn, optimizer, device, train_stats):
    #     """
    #     The training loop for the video classifier, run once per epoch
    #     """
    #     model.train()
    #     train_stats.metrics.reset()
    #     running_loss = 0.0
    #     scaler = GradScaler('cuda')
    #
    #     for images, gt in training_dataloader:
    #         images, gt = images.to(device), gt.type(torch.LongTensor).to(device)
    #         # optimizer.zero_grad()
    #
    #         with autocast('cuda'):
    #             outputs = model(images)
    #             train_stats.metrics.update(outputs, gt - 1)
    #             loss = loss_fn(outputs, gt - 1)
    #
    #         scaler.scale(loss).backward()
    #         scaler.step(optimizer)
    #         scaler.update()
    #         optimizer.zero_grad()
    #
    #         running_loss += loss.item()
    #
    #     avg_train_loss = running_loss / len(training_dataloader)
    #     train_stats.log_metrics()
    #     train_stats.log_loss(avg_train_loss)
    #
    #     # avg_val_loss = evaluate_metrics(model, validation_dataloader, loss_fn,device, train_stats.val_metrics)
    #     # train_stats.log_val_metrics()
    #     # train_stats.log_val_loss(avg_val_loss)
