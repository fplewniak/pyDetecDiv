#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
The central class for keeping track of all available objects in a project.
"""
import subprocess
from typing import Callable, Any, Generator, cast, TypeVar

import json
import os
import itertools
from collections import defaultdict
from datetime import datetime
import pandas as pd
import polars
from PIL import Image
from _polars_runtime_32._polars_runtime import ColumnNotFoundError
from ndtiff import NDTiffDataset

from pydetecdiv.domain.BoundingBox import BoundingBox
from pydetecdiv.domain.Classification import Classification
from pydetecdiv.domain.Entity import Entity
from pydetecdiv.domain.Mask import Mask
from pydetecdiv.domain.Point import Point
from pydetecdiv.domain.RoiAnnotations import RoiAnnotations
from pydetecdiv.settings import get_config_value, Device
from pydetecdiv.persistence.project import open_project
from pydetecdiv.domain.dso import DomainSpecificObject
from pydetecdiv.domain.Dataset import Dataset
from pydetecdiv.domain.Run import Run
from pydetecdiv.domain.ROI import ROI
from pydetecdiv.domain.FOV import FOV
from pydetecdiv.domain.Experiment import Experiment
from pydetecdiv.domain.Data import Data
from pydetecdiv.domain.ImageResource import ImageResource
from pydetecdiv.utils.path import files_in_dir

# TypeVar definitions to enable type checking for subclasses of DomainSpecificObject class
DSO = TypeVar('DSO', bound=DomainSpecificObject)

class Project:
    """
    Project class to keep track of the database connection and providing basic methods to retrieve objects. Actually
    hides repository from other domain classes. This class handles actual records and domain-specific objects while the
    repository deals with records and data-access objects. Data exchange between Repository and Project objects is
    achieved through the use of dict representing standardized records.
    """
    classes = {
        'ROI'          : ROI,
        'FOV'          : FOV,
        'Experiment'   : Experiment,
        'Data'         : Data,
        'Dataset'      : Dataset,
        'ImageResource': ImageResource,
        'Run'          : Run,
        'Entity'       : Entity,
        'BoundingBox'  : BoundingBox,
        'Point'        : Point,
        'Mask'         : Mask,
        'Classification' : Classification,
        'RoiAnnotations' : RoiAnnotations,
        }

    def __init__(self, dbname: str, dbms: str | None = None):
        self.repository = open_project(dbname, dbms)
        self.dbname = dbname
        self.pool = defaultdict(DomainSpecificObject)

    @property
    def path(self) -> str:
        """
        Property returning the path of the project

        :return:
        """
        return os.path.join(get_config_value('project', 'workspace'), self.dbname)

    @property
    def uuid(self) -> str:
        """
        Property returning the uuid associated to this project

        :return:
        """
        return self.get_named_object('Experiment', self.dbname).uuid

    @property
    def author(self) -> str:
        """
        Property returning the author associated to this project

        :return:
        """
        return self.get_named_object('Experiment', self.dbname).author

    @property
    def date(self) -> datetime:
        """
        Property returning the date of this project

        :return:
        """
        return self.get_named_object('Experiment', self.dbname).date

    @property
    def raw_dataset(self) -> Dataset:
        """
        Property returning the raw dataset object associated to this project

        :return:
        """
        return self.get_named_object('Experiment', self.dbname).raw_dataset

    def commit(self) -> None:
        """
        Commit operations performed on objects (creation and update) to save them into the repository.
        """
        self.repository.commit()

    def cancel(self) -> None:
        """
        Cancel operations performed on objects since last commit
        """
        self.repository.rollback()

    def import_images(self, image_files: list[str], destination: str | None = None, **kwargs) -> subprocess.Popen:
        """
        Import images specified in a list of files into a destination

        :param image_files: list of image files to import
        :param destination: destination directory to import files into
        :param kwargs: extra keyword arguments

        :return: the list of imported files. This list can be used to roll the copy back if needed
        """
        data_dir_path = os.path.join(get_config_value('project', 'workspace'), self.dbname, 'data')
        return self.repository.import_images(image_files, data_dir_path, destination, **kwargs)

    def import_images_in_dir(self, image_dir:str, author: str = '', date: str = 'now',
                             img_format: str = 'imagetiff',) -> Generator[int, Any, None]:
        """
        Import all tiff files in directory
        :param image_dir: the directory name
        :param author: the user importing the data
        :param date: the date of import
        :param img_format: the image format
        """
        dataset: Dataset = cast(Dataset, self.get_named_object('Dataset', 'data'))
        author = get_config_value('project', 'user') if author == '' else author
        date_time = datetime.now() if date == 'now' else datetime.fromisoformat(date)
        count = 0

        image_files = files_in_dir(image_dir, ['*.tiff', '*.tif'])
        for image_file in image_files:
            source_dir, rel_url = Device.get_path_id_and_url(image_file)
            with Image.open(image_file) as img:
                width, height = img.size
            _ = Data(project=self, name=os.path.basename(image_file),
                         dataset=dataset, author=author, date=date_time,
                         url=rel_url,
                         format_=img_format, source_dir=source_dir, meta_data={},
                         key_val={}, image_resource=None,
                         c=None, t=None, z=None,
                         xdim=width, ydim=height)
            count += 1
            yield count


    def import_images_from_metadata(self, metadata_files: str, author: str = '', date: str = 'now', img_format: str = 'imagetiff',
                                    resource_format= ImageResource.MULTI) -> Generator[int, Any, None]:
        """
        Import images specified in a list of files into a destination

        :param author: the user id
        :param resource_format: the kind of resource (MULTI, SINGLE, NDTIFF)
        :param img_format: the image format (information stored in the repository)
        :param date: the import date
        :param metadata_files: list of metadata files to load and get information from for image import
        :param destination: destination directory to import image files into (Obsolete)
        :param in_place: if True, image files are not copied (Obsolete)
        :param kwargs: extra keyword arguments
        """
        dataset: Dataset = cast(Dataset, self.get_named_object('Dataset', 'data'))
        author = get_config_value('project', 'user') if author == '' else author
        date_time = datetime.now() if date == 'now' else datetime.fromisoformat(date)
        dirname = os.path.dirname(metadata_files)
        count = 0

        with open(metadata_files) as metadata_file:

            metadata = json.load(metadata_file)
            # positions = [d["Label"] for d in metadata["Summary"]["StagePositions"]]
            sizeT = -1
            fov = None

            for d in [v for k, v in metadata.items() if k.startswith('Metadata-')]:
                if fov is None:
                    fov = FOV(project=self, name=d["PositionName"])
                    image_res = ImageResource(project=self, dataset=dataset, fov=fov, multi=True,
                                              resource_format=resource_format,
                                              zdim=metadata["Summary"]["Slices"],
                                              cdim=metadata["Summary"]["Channels"], tdim=-1,
                                              tscale=metadata["Summary"]["Interval_ms"],
                                              zscale=metadata["Summary"]["z-step_um"],
                                              key_val={'channel_names': metadata["Summary"]["ChNames"]}, )

                image_file = os.path.join(dirname, os.path.basename(str(d["FileName"])))
                source_dir, rel_url = Device.get_path_id_and_url(image_file)

                _ = Data(project=self, name=os.path.basename(image_file),
                         dataset=dataset, author=author, date=date_time,
                         url=rel_url,
                         format_=img_format, source_dir=source_dir, meta_data={},
                         key_val={}, image_resource=image_res,
                         c=d["ChannelIndex"], t=d["FrameIndex"], z=d["SliceIndex"],
                         xdim=d["Width"], ydim=d['Height'])
                maxT = max(sizeT, d["FrameIndex"])
                count += 1
                yield count

            image_res.xdim, image_res.ydim, image_res.tdim = d["Width"], d['Height'], (maxT + 1)
            image_res.validate()

    def import_ndtiff_data(self, ndtiff_dir, author: str = '', date: str = 'now'):
        """
        Import images in NDTiff format

        :param ndtiff_dir: the NDTiff directory
        :param author: the user importing the data
        :param date: the date of import
        """
        dataset: Dataset = cast(Dataset, self.get_named_object('Dataset', 'data'))
        author = get_config_value('project', 'user') if author == '' else author
        date_time = datetime.now() if date == 'now' else datetime.fromisoformat(date)

        ndtiff_ds = NDTiffDataset(ndtiff_dir)
        summary = ndtiff_ds.summary_metadata
        df = polars.DataFrame(ndtiff_ds.get_image_coordinates_list())
        dims_df = df.group_by(by='position').agg(polars.col('time').max(), polars.col('z').max(), polars.col('channel').max())

        tscale = summary["Interval_ms"]
        zscale = summary["z-step_um"]
        channel_names = summary["ChNames"]
        xdim = summary["Width"]
        ydim = summary["Height"]

        count = 0
        for row in dims_df.iter_rows():
            pos_index = row[0]
            tdim = row[1] + 1
            zdim = row[2] + 1
            cdim = row[3] + 1

            fov = FOV(project=self, name=summary["StagePositions"][pos_index]["Label"])

            image_res = ImageResource(project=self, dataset=dataset, fov=fov, multi=False,
                                      resource_format=ImageResource.NDTIFF,
                                      xdim=xdim, ydim=ydim, zdim=zdim, cdim=cdim, tdim=tdim,
                                      xyscale=ndtiff_ds.read_metadata(channel=0, z=0, time=0, position=pos_index)["PixelSizeUm"],
                                      tscale=tscale, zscale=zscale,
                                      key_val={'channel_names': channel_names, 'pos_index': pos_index}
                                      )

            source_dir, rel_url = Device.get_path_id_and_url(ndtiff_dir)

            data = Data(project=self, name=fov.name,
                         dataset=dataset, author=author, date=date_time,
                         url=rel_url,
                         format_='ndtiff', source_dir=source_dir, meta_data={},
                         key_val={}, image_resource=image_res,
                         xdim=xdim, ydim=ydim)

            fov.validate()
            data.validate()
            image_res.validate()
            count += 1
            yield count
        self.commit()

    def annotate(self, dataset: Dataset, source: str | Callable, columns: tuple[str, ...], regex: str) -> pd.DataFrame:
        """
        Annotate data in a dataset using a regular expression applied to columns specified by source (column name or
        callable returning a str built from column names)

        :param dataset: the dataset DSO whose data should be annotated
        :type dataset: Dataset object
        :param source: source column(s) used to determine the annotations
        :type source: str or callable
        :param columns: annotation columns to add to the dataframe representing Data objects
        :type columns: tuple of str
        :param regex: the regular expression defining columns
        :type regex: str
        :return: a table representing annotated Data objects
        :rtype: pandas DataFrame
        """
        return self.repository.annotate_data(dataset, source, columns, regex)

    def create_fov_from_raw_data(self, df: pd.DataFrame, multi: bool) -> Generator[int]:
        """
        Create domain-specific objects from raw data using a regular expression applied to a database field
        or a combination thereof specified by source. DSOs to create are specified by the values in keys.
        """
        yield 0
        fov_names = [f.name for f in self.get_objects('FOV')]
        new_fov_names = df.FOV.drop_duplicates().values
        total = len(new_fov_names) + len(df.values)
        new_fovs = [FOV(project=self, name=fov_name, top_left=(0, 0), bottom_right=(999, 999)) for fov_name in
                    new_fov_names if fov_name not in fov_names]
        if multi:
            _ = {fov.id_: ImageResource(project=self, dataset=self.raw_dataset, fov=fov, multi=True,
                                                          zdim=int(df['Z'].max()),
                                                          cdim=int(df['C'].max()),
                                                          tdim=int(df['T'].max())) for fov in new_fovs}
        else:
            _ = {fov.id_: ImageResource(project=self, dataset=self.raw_dataset, fov=fov, multi=False,
                                                          ) for fov in new_fovs}
        image_resources = {fov.id_: fov.image_resource('data') for fov in self.get_objects('FOV')
                           if fov.image_resource('data').multi}

        yield int(len(new_fov_names) * 100 / total)
        df['FOV'] = df['FOV'].map(self.id_mapping('FOV'))
        # if 'C' in df.columns:
        if multi:
            # for fov_id, image_res in new_image_resources.items():
            for fov_id, image_res in image_resources.items():
                # (image_res.zdim, image_res.cdim, image_res.tdim) = df.loc[df['FOV'] == fov_id, ['Z', 'C', 'T']].astype(
                #         int).max(axis=0).add(1)
                self.save(image_res)
        else:
            # for fov_id, image_res in new_image_resources.items():
            for fov_id, image_res in image_resources.items():
                (image_res.tdim, image_res.cdim, image_res.zdim, image_res._ydim, image_res._xdim,) = image_res.shape
                self.save(image_res)

        for i, (data_id, fov_id) in enumerate(df.loc[:, ['id_', 'FOV']].values):
            data_file = self.get_object('Data', int(data_id))
            data_file.image_resource = self.get_object('FOV', int(fov_id)).image_resource('data').id_
            if multi:
                data_file.c = df.loc[i, 'C']
                data_file.t = df.loc[i, 'T']
                data_file.z = df.loc[i, 'Z']
            self.save(data_file)
            yield int((i + len(new_fov_names)) * 100 / total)

    def id_mapping(self, class_name: str) -> dict[str, int]:
        """
        Return name to id mapping for objects of a given class

        :param class_name: the class name
        :return: the name to id mapping
        """
        return {str(obj.name): cast(int, obj.id_) for obj in self.get_objects(class_name)}

    def save_record(self, class_name: str, record: dict[str, Any]) -> int:
        """
        Creates and saves an object of class named class_name from its record without requiring the creation of a DSO.
        This method can be useful for creating associated objects with mutual dependency.

        :param class_name: the class name of the object to create
        :type class_name: str
        :param record: the record representing the object to create
        :type record: dict
        :return: the id of the created object
        :rtype: int
        """
        id_ = self.repository.save_object(class_name, record)
        return id_

    def save(self, dso: DSO) -> int:
        """
        Save an object to the repository if it is new or has been modified

        :param dso: the domain-specific object to save
        :return: the id of the saved object if it was created
        """
        id_ = self.repository.save_object(dso.__class__.__name__, dso.record())
        if (dso.__class__.__name__, id_) not in self.pool:
            self.pool[(dso.__class__.__name__, id_)] = dso
        return id_

    def delete(self, dso: DSO) -> None:
        """
        Delete a domain-specific object

        :param dso: the object to delete
        :type dso: object (DomainSpecificObject)
        """
        if dso is not None and dso.id_ is not None:
            if (dso.__class__.__name__, dso.id_) in self.pool:
                del self.pool[dso.__class__.__name__, dso.id_]
            self.repository.delete_object(dso.__class__.__name__, dso.id_)

    def get_object(self, class_name: str, id_: int | None = None, uuid: str | None = None, use_pool: bool = True) -> Any:
        """
        Get an object referenced by its id

        :param class_name: the class of the requested object
        :param id_: the id reference of the object
        :type class_name: class inheriting DomainSpecificObject
        :type id_: int
        :param uuid: the uuid of the requested object
        :param use_pool: True if object should be obtained from the pool unless it has not been created yet
        :return: the desired object
        """
        record = self.repository.get_record(class_name, id_, uuid)
        if record is not None:
            return self.build_dso(class_name, record, use_pool)
        return None

    def get_named_object(self, class_name, name=None, use_pool: bool = True) -> DSO | None:
        """
        Return a named object by its name

        :param class_name: class name of the requested object
        :param name: the name of the requested object
        :return: the object
        """
        record = self.repository.get_record_by_name(class_name, name)
        if record is not None:
            return self.build_dso(class_name, record, use_pool)
        return None

    def get_objects(self, class_name: str, id_list: list[int] | None = None) -> list[DSO | None]:
        """
        Get a list of all domain objects of a given class in the current project retrieved from the repository

        :param class_name: the class name of the domain-specific objects to be returned
        :type class_name: str
        :param id_list: the list of ids for the objects to be retrieved
        :type id_list: list of int
        :return: a list of all the objects of that class in the project with all their associated metadata
        :rtype: list of the requested domain-specific objects
        """
        if class_name == 'ROI':
            return self._get_rois(id_list)
        records = self.repository.get_records(class_name, id_list)
        if records is not None:
            return [self.build_dso(class_name, rec) for rec in records]
        return []

    def get_records(self, class_name: str, id_list: list[int] | None = None) -> list[dict[str, Any]]:
        """
        get list of dictionary records of DSOs of the specified class with id in list
        :param class_name: the name of the class
        :param id_list: the list of ids
        """
        return self.repository.get_records(class_name, id_list)

    def get_dataframe(self, class_name: str, id_list: list[int] | None = None) -> pd.DataFrame:
        """
        get a pandas DataFrame with records of DSOs of the specified class with id in list
        :param class_name: the name of the class
        :param id_list: the list of ids
        """
        return pd.DataFrame.from_records(self.get_records(class_name, id_list))

    def get_polars(self, class_name: str, id_list: list[int] | None = None) -> polars.DataFrame:
        """
        Returns a list of objects of a certain class with their records presented as a polars.DataFrame

        :param class_name: the name of the class
        :param id_list: a list of ids to restrict the output
        :return: the polars.DataFrame presenting the requested objects
        """
        records = self.get_records(class_name, id_list)
        if records is not None:
            df = polars.from_records(records)
            try:
                df.get_column('key_val')
                return df.unnest('key_val')
            except ColumnNotFoundError:
                return df
        return polars.DataFrame(schema=self.repository.get_field_names(class_name))

    def count_objects(self, class_name: str) -> int:
        """
        Count all objects of a given class in the current project

        :param class_name: the class name of the domain-specific objects to count
        :type class_name: str
        :return: the number of objects of the requested class
        :rtype: int
        """
        return self.repository.count_records(class_name)

    def get_annotated_rois(self, ids_only=False, id_list: list[int] | None = None) -> list[ROI] | list[int]:
        """
        Get a list of ROIs, or ids thereof, which have annotations

        :param ids_only: return ids if True, actual ROI objects otherwise
        :param id_list: a list of ROI ids to restrict the list of returned ROIs
        :return: the list of annotated ROIs or ids thereof
        """
        annotations_df = self.get_polars('RoiAnnotations')
        roi_ids = annotations_df.select('roi').unique().to_series().to_list()
        if id_list is not None:
            roi_ids = list(set(roi_ids).intersection(set(id_list)))
        if ids_only:
            return sorted(roi_ids)
        return [cast(ROI, roi) for roi in self.get_objects('ROI', roi_ids)]

    def _get_rois(self, id_list: list[int] | None = None) -> list[ROI]:
        """
        Gets ROIs using FOV.roi_list properties for all FOVs in order to show the initial ROIs only for FOVs that have
        no other defined ROI. This method is used by the generic get_objects method to deal with the special case of
        initial ROIs

        :param id_list: the list of ids for the ROIs to be retrieved
        :type id_list: list of int
        :return: a list of the requested ROIs
        :rtype: list of ROI objects
        """
        all_rois = itertools.chain(*[fov.roi_list for fov in self.get_objects('FOV')])
        if id_list is None:
            return list(all_rois)
        return [roi for roi in all_rois if roi.id_ in id_list]

    def count_orphan_data_files(self) -> int:
        """
        Count image files that are not associated with an image resource
        """
        return self.repository.count_orphan_data_files()

    def has_links(self, class_name: str, to: DSO) -> bool:
        """
        Checks whether there are links to a given object from objects of a given class.

        :param class_name: the name of the class from which the existence of links is tested
        :type class_name: str
        :param to: the object which is tested for existence of links from class_name class
        :type to: DomainSpecificObject object
        :return: True if links exist, False otherwise
        :rtype: bool
        """
        if to.id_ is not None and self.repository.get_linked_records(class_name, to.__class__.__name__, to.id_):
            return True
        return False

    def count_links(self, class_name: str, to: DSO) -> int:
        """
        Counts the number of objects of a given class having a link to an object

        :param class_name: the name of the class whose links to the specified object are counted
        :type class_name: str
        :param to: the object to which links are counted
        :type to: DomainSpecificObject object
        :return: the number of objects of class class_name that are linked to the specified object
        :rtype: int
        """
        if to.id_ is not None:
            return len(self.repository.get_linked_records(class_name, to.__class__.__name__, to.id_))
        return 0

    def get_linked_objects(self, class_name: str, to: DSO) -> list[Any]:
        """
        A method returning the list of all objects of class defined by class_name that are linked to an object specified
        by argument to=
        Note that for ROIs linked to a FOV, this method also returns the initial ROI (full-FOV) even if other ROIs have
        been created. If initial ROIs are not desired, then the FOV.roi_list property should be used instead.

        :param class_name: the class name of the objects to retrieve
        :type class_name: str
        :param to: the object the retrieve objects should be linked to
        :type to: DomainSpecificObject
        :return: the list of objects linked to linked_to object
        :rtype: list of objects
        """
        object_list = []
        if to.id_ is not None:
            object_list = [self.build_dso(class_name, rec) for rec in
                           self.repository.get_linked_records(class_name, to.__class__.__name__, to.id_)]
        return object_list

    def get_linked_records(self, class_name: str, to: DSO) -> list[dict[str, Any]]:
        """
         A method returning the list of all records of class defined by class_name that are linked to an object specified
        by argument to=

        :param class_name: the class name of the objects to retrieve
        :type class_name: str
        :param to: the object the retrieve objects should be linked to
        :type to: DomainSpecificObject
        :return: the list of records linked to linked_to object
        """
        if to.id_ is not None:
            return self.repository.get_linked_records(class_name, to.__class__.__name__, to.id_)
        return []

    def link_objects(self, dso1: DSO, dso2: DSO) -> None:
        """
        Create a direct link between two objects. This method only works for objects that have a direct logical
        connection defined in an association table. It does not work to create transitive links with intermediate
        objects

        :param dso1: first domain-specific object to link
        :type dso1: object
        :param dso2: second domain-specific object to link
        :type dso2: object
        """
        if dso1.id_ is not None and dso2.id_ is not None:
            self.repository.link(dso1.__class__.__name__, dso1.id_, dso2.__class__.__name__, dso2.id_, )

    def unlink_objects(self, dso1: DSO, dso2: DSO) -> None:
        """
        Delete a direct link between two objects. This method only works for objects that have a direct logical
        connection defined in an association table. It does not work to delete transitive links with intermediate
        objects

        :param dso1: first domain-specific object to unlink
        :type dso1: object
        :param dso2: second domain-specific object to unlink
        :type dso2: object
        """
        if dso1.id_ is not None and dso2.id_ is not None:
            self.repository.unlink(dso1.__class__.__name__, dso1.id_, dso2.__class__.__name__, dso2.id_, )

    def build_dso(self, class_name: str, rec: dict[str, Any] | None, use_pool: bool = True) -> DSO | None:
        """
        factory method to build a dso of class class_name from record rec or return the pooled object if it was already
        created. Note that if the object was already in the pool, values in the record are not used to update the
        object. To make any change to the object's attribute, the object's class methods and setter should be used
        instead as they ensure that values are checked for validity and new attributes are saved to the persistence
        back-end

        :param class_name: the class name of the dso to build
        :param rec: the record representing the object to build (if id_ is not in the record, the object is a new creation
        :param use_pool: use the project's object pool if True
        :return: the requested object
        """
        if rec is None:
            return None
        if 'id_' in rec and (class_name, rec['id_']) in self.pool:
            if use_pool:
                return self.pool[(class_name, rec['id_'])]
        obj = Project.classes[class_name](project=self, **rec)
        self.pool[(class_name, obj.id_)] = obj
        return obj

    @property
    def info(self) -> str:
        """
        Returns ready-to-print information about project

        :return: project information
        :rtype: str
        """
        return f"""
Name:               {self.dbname}
Author:             {self.author}
Date:               {self.date}
number of FOV:      {len(self.get_objects('FOV'))}
number of ROI:      {len(self.get_objects('ROI'))}
number of datasets: {len(self.get_objects('Dataset'))}
number of files:    {len(self.get_objects('Data'))}
number of runs:     {len(self.get_objects('Run'))}
        """
