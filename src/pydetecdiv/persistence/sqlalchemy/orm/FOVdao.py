#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Access to FOV data
"""
from sqlalchemy import Column, Integer, String, text, select
from sqlalchemy.types import JSON
from sqlalchemy.orm import relationship, joinedload
from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base


class FOVdao(DAO, Base):
    """
    DAO class for access to FOV records from the SQL database
    """
    __tablename__ = 'FOV'
    exclude = ['id_', 'top_left', 'bottom_right']
    # translate = {'size': ('xsize', 'ysize'), }
    translate = {}

    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    uuid = Column(String(36))
    name = Column(String, unique=True, nullable=False)
    comments = Column(String)
    key_val = Column(JSON)

    # xsize = Column(Integer, nullable=False, server_default=text('1000'))
    # ysize = Column(Integer, nullable=False, server_default=text('1000'))

    roi_list_ = relationship('ROIdao')

    # data_list = FovData.fov_to_data()

    image_resources_ = relationship('ImageResourceDao')

    def image_resources(self, fov_id: int) -> list[dict[str, object]]:
        """
        A method returning the list of ImageResource records whose parent FOV has id\_ == fov_id

        :param fov_id: the id of the FOV
        :return: a list of ImageResource records whose parent FOV has id\_ == fov_id
        """
        if self.session.query(FOVdao).filter(FOVdao.id_ == fov_id).first() is not None:
            data_list = [data.record
                         for data in self.session.query(FOVdao)
                         .options(joinedload(FOVdao.image_resources_))
                         .filter(FOVdao.id_ == fov_id)
                         .first().image_resources_]
        else:
            data_list = []
        return data_list

    @property
    def record(self) -> dict[str, object]:
        """
        A method creating a DAO record dictionary from a fov row dictionary. This method is used to convert the SQL
        table columns into the FOV record fields expected by the domain layer

        :return: a FOV record as a dictionary with keys() appropriate for handling by the domain layer
        """
        return {'id_': self.id_,
                'name': self.name,
                'comments': self.comments,
                'uuid': self.uuid,
                'key_val': self.key_val,
                }

    # def data(self, fov_id):
    #     """
    #     Returns a list of DataDao objects linked to the FOVdao object with the specified id_
    #
    #     :param fov_id: the id_ of the FOV
    #     :type fov_id: int
    #     :return: the list of Data records linked to the FOV
    #     :type: list of dict
    #     """
    #     return [i.record
    #             for i in self.session.query(dao.DataDao)
    #             .filter(FovData.data == dao.DataDao.id_)
    #             .filter(FovData.fov == fov_id)
    #     ]

    # def image_data(self, fov_id):
    #     """
    #     A method returning the list of Image data object records linked to FOV with id_ == fov_id
    #     :param fov_id: the id of the FOV
    #     :type fov_id: int
    #     :return: a list of ImageData records linked to FOV with id_ == fov_id
    #     :rtype: list
    #     """
    #     image_data = [i.record
    #                   for i in itertools.chain(*[roi.image_data_list
    #                                              for roi in self.session.query(dao.ROIdao)
    #                                            .filter(dao.ROIdao.fov == fov_id)
    #                                            .all()])]
    #     return image_data

    def roi_list(self, fov_id: int) -> list[dict[str, object]]:
        """
        A method returning the list of ROI records whose parent FOV has id == fov_id

        :param fov_id: the id of the FOV
        :return: a list of ROI records with parent FOV id == fov_id
        """
        if self.session.query(FOVdao).filter(FOVdao.id_ == fov_id).first() is not None:
            roi_list = [roi.record
                        for roi in self.session.query(FOVdao)
                        .options(joinedload(FOVdao.roi_list_))
                        .filter(FOVdao.id_ == fov_id)
                        .first().roi_list_]
        else:
            roi_list = []
        return roi_list

    # def image_list(self, fov_id):
    #     """
    #     A method returning the Image records linked to FOV with id_ == fov_id
    #     :param fov_id: the id of the Image data
    #     :type fov_id: int
    #     :return: a list containing the Image records linked to FOV with id_ == fov_id
    #     :rtype: list
    #     """
    #     image_list = [image.record for image in
    #                   self.session.scalars(
    #                       select(dao.ImageDao)
    #                       .where(dao.ImageDataDao.id_ == dao.ImageDao.image_data)
    #                       .where(dao.ROIdao.id_ == dao.ImageDataDao.roi)
    #                       .where(fov_id == dao.ROIdao.fov)
    #                   )]
    #     return image_list
