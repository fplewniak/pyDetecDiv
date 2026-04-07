import tables


class TblShortNamesRow(tables.IsDescription):
    """
    A class to describe the class names row saved in a table of an HDF5 file
    """
    name = tables.StringCol(8)


class TblNamesRow(tables.IsDescription):
    """
    A class to describe the class names row saved in a table of an HDF5 file
    """
    name = tables.StringCol(16)


class TblLongNamesRow(tables.IsDescription):
    """
    A class to describe the class names row saved in a table of an HDF5 file
    """
    name = tables.StringCol(32)


class TblVeryLongNamesRow(tables.IsDescription):
    """
    A class to describe the class names row saved in a table of an HDF5 file
    """
    name = tables.StringCol(128)
