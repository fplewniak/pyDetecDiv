CREATE TABLE Dataset (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
);

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

CREATE TABLE FOVdata (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    fov       INTEGER REFERENCES FOV (id)
                      NOT NULL,
    imagedata INTEGER REFERENCES ImageData (id) ON DELETE CASCADE
                      NOT NULL
);

CREATE TABLE FOVprocess (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    fov        INTEGER REFERENCES FOV (id) ON DELETE CASCADE
                       NOT NULL,
    processing INTEGER REFERENCES Processing (id) ON DELETE CASCADE
                       NOT NULL
);

CREATE TABLE History (
    id   INTEGER  PRIMARY KEY AUTOINCREMENT,
    time DATETIME NOT NULL
);

CREATE TABLE HistoryFOVprocess (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    history    INTEGER REFERENCES History (id) ON DELETE CASCADE,
    fovprocess INTEGER REFERENCES FOVprocess (id) ON DELETE CASCADE
);

CREATE TABLE HistoryROIprocess (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    history    INTEGER REFERENCES History (id) ON DELETE CASCADE
                       NOT NULL,
    roiprocess INTEGER REFERENCES ROIprocess (id) ON DELETE CASCADE
                       NOT NULL
);

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

CREATE TABLE ImageData (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT    DEFAULT ('original capture'),
    channel   INTEGER NOT NULL
                      DEFAULT (0),
    xsize     INTEGER NOT NULL
                      DEFAULT (1000),
    ysize     INTEGER DEFAULT (1000)
                      NOT NULL,
    zsize     INTEGER NOT NULL
                      DEFAULT (1),
    tsize     INTEGER NOT NULL
                      DEFAULT (1),
    interval  TIME    DEFAULT (0),
    orderdims TEXT    NOT NULL
                      DEFAULT ('xyzct'),
    resource  INTEGER REFERENCES ImageResource (id),
    path      TEXT
);

CREATE TABLE ImageDataset (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset INTEGER REFERENCES Dataset (id) ON DELETE CASCADE,
    image   INTEGER REFERENCES Image (id) ON DELETE CASCADE
);

CREATE TABLE ImageResource (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    locator TEXT    NOT NULL
);

CREATE TABLE Processing (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    processor  INTEGER REFERENCES Processor (id),
    parameters TEXT
);

CREATE TABLE Processor (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    type        TEXT    NOT NULL,
    description TEXT
);

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

CREATE TABLE ROIdata (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    roi       INTEGER REFERENCES ROI (id) ON DELETE CASCADE
                      NOT NULL,
    imagedata INTEGER NOT NULL
                      REFERENCES ImageData (id) ON DELETE CASCADE
);

CREATE TABLE ROIprocess (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    roi        INTEGER REFERENCES ROI (id) ON DELETE CASCADE
                       NOT NULL,
    processing INTEGER REFERENCES Processing (id) ON DELETE CASCADE
                       NOT NULL
);


