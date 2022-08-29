--
-- File generated with SQLiteStudio v3.3.3 on lun. août 22 11:53:58 2022
--
-- Text encoding used: UTF-8
--
PRAGMA foreign_keys = off;
BEGIN TRANSACTION;

-- Table: Dataset
CREATE TABLE Dataset (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
);


-- Table: FileResource
CREATE TABLE FileResource (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    locator  TEXT    NOT NULL,
    mimetype TEXT    NOT NULL
);


-- Table: FOV
CREATE TABLE FOV (
    id       INTEGER PRIMARY KEY AUTOINCREMENT
                     UNIQUE,
    name     TEXT    UNIQUE,
    comments TEXT,
    xsize    INTEGER NOT NULL
                     DEFAULT (1000),
    ysize    INTEGER NOT NULL
                     DEFAULT (1000) 
);


-- Table: FOVdata
CREATE TABLE FOVdata (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    fov       INTEGER REFERENCES FOV (id) 
                      NOT NULL,
    imagedata INTEGER REFERENCES ImageData (id) ON DELETE CASCADE
                      NOT NULL
);


-- Table: FOVprocess
CREATE TABLE FOVprocess (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    fov        INTEGER REFERENCES FOV (id) ON DELETE CASCADE
                       NOT NULL,
    processing INTEGER REFERENCES Processing (id) ON DELETE CASCADE
                       NOT NULL
);


-- Table: History
CREATE TABLE History (
    id   INTEGER  PRIMARY KEY AUTOINCREMENT,
    time DATETIME NOT NULL
);


-- Table: HistoryFOVprocess
CREATE TABLE HistoryFOVprocess (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    history    INTEGER REFERENCES History (id) ON DELETE CASCADE,
    fovprocess INTEGER REFERENCES FOVprocess (id) ON DELETE CASCADE,
    results    INTEGER REFERENCES Results (id) 
);


-- Table: HistoryROIprocess
CREATE TABLE HistoryROIprocess (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    history    INTEGER REFERENCES History (id) ON DELETE CASCADE
                       NOT NULL,
    roiprocess INTEGER REFERENCES ROIprocess (id) ON DELETE CASCADE
                       NOT NULL,
    results    INTEGER REFERENCES Results (id) 
);


-- Table: Image
CREATE TABLE Image (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    z      INTEGER NOT NULL
                   DEFAULT (0),
    t      INTEGER NOT NULL
                   DEFAULT (0),
    xdrift INTEGER,
    ydrift INTEGER,
    data   INTEGER REFERENCES ImageData (id) 
);


-- Table: ImageData
CREATE TABLE ImageData (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT    DEFAULT ('original capture'),
    channel   INTEGER NOT NULL
                      DEFAULT (0),
    x0        INTEGER NOT NULL
                      DEFAULT (0),
    y0        INTEGER DEFAULT (0) 
                      NOT NULL,
    x1        INTEGER NOT NULL
                      DEFAULT ( -1),
    y1        INTEGER NOT NULL
                      DEFAULT ( -1),
    zsize     INTEGER NOT NULL
                      DEFAULT (1),
    tsize     INTEGER NOT NULL
                      DEFAULT (1),
    interval  TIME    DEFAULT (0),
    orderdims TEXT    NOT NULL
                      DEFAULT ('xyzct'),
    resource  INTEGER REFERENCES FileResource (id),
    path      TEXT,
    mimetype  TEXT
);


-- Table: ImageDataset
CREATE TABLE ImageDataset (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset INTEGER REFERENCES Dataset (id) ON DELETE CASCADE,
    image   INTEGER REFERENCES Image (id) ON DELETE CASCADE
);


-- Table: Processing
CREATE TABLE Processing (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    processor  INTEGER REFERENCES Processor (id),
    parameters TEXT
);


-- Table: Processor
CREATE TABLE Processor (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    type        TEXT    NOT NULL,
    description TEXT
);


-- Table: ResultData
CREATE TABLE ResultData (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    results  INTEGER REFERENCES Results (id),
    resource INTEGER REFERENCES FileResource (id),
    path     TEXT,
    mimetype TEXT    NOT NULL
);


-- Table: ResultImageData
CREATE TABLE ResultImageData (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    results   INTEGER REFERENCES Results (id),
    imagedata INTEGER REFERENCES ImageData (id) 
);


-- Table: Results
CREATE TABLE Results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    classification TEXT,
    freetext       TEXT
);


-- Table: ROI
CREATE TABLE ROI (
    id   INTEGER PRIMARY KEY AUTOINCREMENT
                 UNIQUE,
    name TEXT,
    fov  INTEGER REFERENCES FOV (id) ON DELETE SET NULL,
    x0   INTEGER DEFAULT (0),
    y0   INTEGER DEFAULT (0),
    x1   INTEGER DEFAULT ( -1),
    y1   INTEGER DEFAULT ( -1) 
);


-- Table: ROIdata
CREATE TABLE ROIdata (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    roi       INTEGER REFERENCES ROI (id) ON DELETE CASCADE
                      NOT NULL,
    imagedata INTEGER NOT NULL
                      REFERENCES ImageData (id) ON DELETE CASCADE
);


-- Table: ROIprocess
CREATE TABLE ROIprocess (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    roi        INTEGER REFERENCES ROI (id) ON DELETE CASCADE
                       NOT NULL,
    processing INTEGER REFERENCES Processing (id) ON DELETE CASCADE
                       NOT NULL
);


COMMIT TRANSACTION;
PRAGMA foreign_keys = on;
