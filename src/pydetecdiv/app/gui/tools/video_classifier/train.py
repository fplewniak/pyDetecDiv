"""
GUI classes for video classifier model training
"""
from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.gui.core.widgets import set_connections
from pydetecdiv.app.gui.tools import ToolDialog
from pydetecdiv.app.gui.tools.deep_learning import plot_training_results
from pydetecdiv.app.parameters import Parameter
from pydetecdiv.app.tools.deep_learning import DeepTool



class TrainModelDialog(ToolDialog):
    """
    Dialog to choose the parameters for training a Video classifier model
    """

    def __init__(self, tool: DeepTool, **kwargs):
        super().__init__(tool, title='Training Video classifier', **kwargs)

        self.model_choice = self.addGroupBox(title='Model',
                                             parameters=[
                                                 tool.parameters.model,
                                                 tool.parameters.layers,
                                                 tool.parameters.strides,
                                                 tool.parameters.dropout,
                                                 ])

        self.hyperparameters = self.addGroupBox(title='Hyperparameters', expandable=True, show=True,
                                                parameters=[
                                                    tool.parameters.epochs,
                                                    tool.parameters.batch_size,
                                                    tool.parameters.optimizer,
                                                    tool.parameters.learning_rate,
                                                    tool.parameters.focal_gamma,
                                                    tool.parameters.seed,
                                                    ],
                                                widget_args={
                                                    'learning_rate': {'adaptive': True, },
                                                    }
                                                )

        self.scheduler = self.addGroupBox(title='Scheduler', expandable=True, show=False,
                                          parameters=[
                                              tool.parameters.step_scheduler,
                                              tool.parameters.step_gamma,
                                              tool.parameters.step_size,
                                              tool.parameters.warmup,
                                              tool.parameters.wu_start,
                                              tool.parameters.wu_end,
                                              tool.parameters.wu_duration,
                                              tool.parameters.reduce_lr_on_plateau,
                                              tool.parameters.reduce_patience,
                                              tool.parameters.reduction_factor,
                                              ],
                                          )

        self.regularization = self.addGroupBox(title='Regularization', expandable=True, show=False,
                                               parameters=[
                                                   tool.parameters.regularization,
                                                   tool.parameters.lambda_reg,
                                                   ],
                                               widget_args={
                                                   'lambda_reg': {'adaptive': True},
                                                   }
                                               )

        self.datasets = self.addGroupBox(title='Datasets', expandable=True, show=False,
                                         parameters=[
                                             tool.parameters.num_training,
                                             tool.parameters.num_validation,
                                             tool.parameters.num_test,
                                             tool.parameters.data_seed,
                                             tool.parameters.augmentation,
                                             tool.parameters.seq_len,
                                             # tool.parameters.idx,
                                             ],
                                         widget_args={
                                             'num_test': {'enabled': False},
                                             }
                                         )

        self.tool.parameters.hdf5_file.current_dir = PyDetecDiv.tools['cnrs.plewniak.roiseqhdf5creator'].working_dir
        self.hdf5_file = self.addGroupBox(title='Data File',
                                          parameters=[
                                              tool.parameters.hdf5_file,
                                              tool.parameters.time_first,
                                              ],
                                          )

        self.button_box = self.addButtonBox()

        self.arrangeWidgets([self.model_choice,
                             self.hyperparameters,
                             self.regularization,
                             self.scheduler,
                             self.datasets, self.hdf5_file,
                             self.button_box])

        set_connections({self.button_box.accepted              : lambda: self.wait_for_process(tool.model_trainer.train_model,
                                                                                               '**Training model**'),
                         self.button_box.rejected              : lambda: print('Rejected'),
                         tool.parameters.num_training.changed  : lambda: self.update_datasets(tool.parameters.num_training),
                         tool.parameters.num_validation.changed: lambda: self.update_datasets(tool.parameters.num_validation),
                         })

        self.run_after_process([plot_training_results, tool.dump_train_stats])

        # tool.parameters.reset()
        self.fit_to_contents()
        self.exec()

    def update_datasets(self, changed_param: Parameter) -> None:
        """
        Update the dataset proportions values to make sure they sup up to 1

        :param changed_param: the changed parameter
        """
        self.tool.parameters.num_test.value = 1.0 - (self.tool.parameters.num_training + self.tool.parameters.num_validation)
        if changed_param:
            total = self.tool.parameters.num_test + self.tool.parameters.num_training + self.tool.parameters.num_validation
            if total > 1.0:
                changed_param.value = changed_param - total + 1.0
