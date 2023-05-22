import bioimageit_core.containers


class DatasetInfo(bioimageit_core.containers.data_containers.DatasetInfo):
    def __init__(self, rds_name, rds, rds_uuid):
        super().__init__(rds_name, rds, rds_uuid)
        self.id_ = None
