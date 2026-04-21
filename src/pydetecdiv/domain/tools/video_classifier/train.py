"""
Video classifier trainer class
"""
import os
from datetime import datetime
from typing import TYPE_CHECKING

import polars
import torch
from torch import GradScaler, autocast
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.transforms import v2, InterpolationMode

from pydetecdiv.app import pydetecdiv_project, PyDetecDiv, get_project_dir
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'running training on {"GPU" if device.type == "cuda" else "CPU"}')

        torch.random.manual_seed(self.tool.parameters.seed.value)

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

        training_dataloader = DataLoader(training_dataset, batch_size=self.tool.parameters.batch_size.value, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=self.tool.parameters.batch_size.value, shuffle=True)

        print(class_weights)
        print(training_dataset.class_names)
        loss_fn = FocalLoss(alpha=class_weights, gamma=1.0, reduction='mean')

        # summary(model, (self.tool.parameters['batch_size'].value, 15, 3, 224, 224), device=device)

        run = self.tool.save_run(command='train_model')
        print(run)

        base_checkpoint_path = os.path.join(get_project_dir(), 'video_classification', 'checkpoints', 'runs', str(run.id_))
        os.makedirs(base_checkpoint_path, exist_ok=True)

        for epoch in range(self.tool.parameters['epochs'].value):
            self.training_loop(training_dataloader, validation_dataloader, model, loss_fn, optimizer, device, train_stats)
            print(f"Epoch {epoch + 1}/{self.tool.parameters['epochs'].value}, "
                  f"Training Loss: {train_stats.history.loss[-1]:.4f}, "
                  f"Validation Loss: {train_stats.history.val_loss[-1]:.4f}, "
                  f"{main_metric}: {train_stats.history.metric_history(main_metric)[-1]:.3f}, "
                  f"Val {main_metric}: {train_stats.history.val_metric_history(main_metric)[-1]:.3f}, "
                  # f"learning rate: {scheduler.get_last_lr()[0]:0.2e}, "
                  f" -- ({datetime.now().strftime('%H:%M:%S')})")

            if train_stats.is_best_val_loss(epoch):
                checkpoint_filepath = os.path.join(base_checkpoint_path, f'epoch{epoch}_best_loss.pt')
                model_scripted = torch.jit.script(model)
                model_scripted.save(checkpoint_filepath)
                print(f"Saving best model at epoch {epoch + 1} with val loss {train_stats.history.val_loss[-1]:.4f}"
                      f" and train loss {train_stats.history.loss[-1]:.4f}")

        checkpoint_filepath = os.path.join(base_checkpoint_path, f'last_epoch{epoch}.pt')
        model_scripted = torch.jit.script(model)
        model_scripted.save(checkpoint_filepath)

        # idx = self.tool.parameters.idx.value
        # idx = 0
        # sequence, target = training_dataset[idx]
        # print(sequence.shape)
        # roi_id, frame = training_dataset.get_ref(idx)
        # with pydetecdiv_project(PyDetecDiv.project_name) as project:
        #     roi = project.get_object('ROI', roi_id)
        #     print(roi.annotations()[frame])
        # print(training_dataset.indices[idx - 5: idx + 5])
        # print(polars.DataFrame([{'roi': training_dataset.get_ref(i)[0],
        #                           'frame': training_dataset.get_ref(i)[1],
        #                           'target': training_dataset[i][1],
        #                          'class': training_dataset.class_names[training_dataset[i][1]]} for i in range(idx - 5, idx + 5)]))
        # rowlen = 5
        # plot_viewer = MatplotViewer(PyDetecDiv.main_window.active_subwindow, columns=rowlen, rows=3)
        # for i in range(3):
        #     for j in range(rowlen):
        #         img_channel_last = torch.as_tensor(sequence[rowlen * i + j].permute([1, 2, 0]))
        #         plot_viewer.axes[i][j].imshow(img_channel_last)
        #         if (rowlen * i + j) == int(3 * rowlen / 2):
        #             plot_viewer.axes[i][j].set_title(f'{training_dataset.class_names[target]}')
        #         plot_viewer.axes[i][j].set_xlabel(f'{frame + rowlen * i + j}')
        # tab = PyDetecDiv.main_window.add_tabbed_window(f'{PyDetecDiv.project_name} / {roi_id}')
        # tab.project_name = PyDetecDiv.project_name
        # tab.addTab(plot_viewer, 'Sample sequence')
        # tab.setCurrentWidget(plot_viewer)

        training_dataset.close()
        validation_dataset.close()
