import glob
from collections import namedtuple
import re
import os

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    'foreground',
    ])
class DatasetConfig(object):
    labels = None
    root = None
    output_dir = 'tfrecord'
    num_shards=None
    img_size = None
    folders_map = {
        'image': '',
        'label': '',
    }
    # A map from data type to filename postfix.
    postfix_map = {
        'image': '',
        'label': '',
    }

    # A map from data type to data format.
    data_format = {
        'image': 'png',
        'label': 'png',
    }
    dataset_split = None

    # Image file pattern.
    image_file_re = re.compile('(.+)' + postfix_map['image'])
    def __init__(self):
        pass

    @property
    def abs_tfrecord_path(self):
        return os.path.abspath(os.path.join(self.root,self.output_dir))

CITYSCAPES=DatasetConfig()
CITYSCAPES.labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color              foreground
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ,  0),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ,  0),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ,  0),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ,0),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ,0),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ,0),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ,0),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ,0),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ,0),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ,0),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ,0),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ,0),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ,0),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ,0),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ,0),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ,0),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ,0),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ,0),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ,0),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ,1),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ,1),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ,0),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ,0),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ,0),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ,1),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ,1),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ,1),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ,1),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ,1),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ,0),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ,0),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ,1),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ,1),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ,1),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ,0),
]
CITYSCAPES.root='H:\PYTHON PROJECTS\ImageSegProject\dataset\CityScapes'
CITYSCAPES.outpur_dir='tfrecord'
CITYSCAPES.num_shards = 10
CITYSCAPES.img_size= {
    'image': (1024,2048,3),
    'label': (1024,2048,1),
    'trainLabel': (1024,2048,1),
    'foregroundLabel': (1024,2048,1),
}

CITYSCAPES.postfix_map = {
    'image': '_leftImg8bit',
    'label': '_gtFine_labelIds',
    'trainLabel': '_labelTrainIds',
    'foregroundLabel': '',
}
CITYSCAPES.data_format = {
    'image': 'png',
    'label': 'png',
    'trainLabel': 'png',
    'foregroundLabel': 'png',

}
CITYSCAPES.dataset_split = {
        'train': 'train',
        'val': 'val',
}
CITYSCAPES.folders_map = {
        'image': 'leftImg8bit',
        'label': 'gtFine',
        'trainLabel':'trainLabels',
        'foregroundLabel': 'forgroundLabel',
}
CITYSCAPES.dataset_size= {
    'train': 2975,
    'val': 500,
}

SYNTHIA_CS=DatasetConfig()
SYNTHIA_CS.labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'void'                 ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0),0 ),
    Label(  'sky'                  ,  1 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180),0 ),
    Label(  'building'             ,  2 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70),0 ),
    Label(  'road'                 ,   3,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ,0),
    Label(  'sidewalk'             ,  4 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ,0),
    Label(  'fence'                ,  5 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153),0 ),
    Label(  'vegetation'           , 6 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ,0),
    Label(  'pole'                 , 7 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ,0),
    Label(  'car'                  , 8 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ,1),
    Label(  'traffic sign'         , 9 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ,1),
    Label(  'person'               , 10 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ,1),
    Label(  'bicycle'              , 11 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ,1),
    Label(  'motorcycle'           , 12 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ,1),
    Label(  'Parking-slot'           ,13 ,     255 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ,0),
    Label(  'road-work'           ,   14 ,     255 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ,0),
    Label(  'traffic light'        , 15 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ,1),
    Label(  'terrain'              , 16 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ,0),
    Label(  'rider'                , 17 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ,1),
    Label(  'truck'                , 18 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70),1 ),
    Label(  'bus'                  , 19 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ,1),
    Label(  'train'                , 20 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ,1),
    Label('wall',                     21,        3, 'construction',     2,        False,          False,        (102, 102, 156),0),
    Label('Lanemarking',             22,       255, 'object',           2,        False,          False,        (102, 102, 156),0),
]
SYNTHIA_CS.root='H:\PYTHON PROJECTS\ImageSegProject\dataset\SYNTHIA_CS'
SYNTHIA_CS.output_dir='tfrecord'
SYNTHIA_CS.num_shards = 10
SYNTHIA_CS.img_size= {
    'image': (760, 1280, 3),
    'label': (760, 1280, 3),
    'trainLabel': (760, 1280, 1),
     'foregroundLabel':(760, 1280, 1),
}
SYNTHIA_CS.postfix_map = {
    'image': '',
    'label': '',
    'trainLabel': '',
'foregroundLabel':'',

}
SYNTHIA_CS.data_format = {
    'image': 'png',
    'label': 'png',
    'trainLabel': 'png',
'foregroundLabel':'png',
}
SYNTHIA_CS.folders_map = {
        'image': 'RGB',
        'label': 'LABELS',
        'trainLabel':'trainLabels',
        'foregroundLabel':'foregroundLabel',

}
SYNTHIA_CS.dataset_size= {
    'train': 7520,
    'val': 1880,
}

class SearchFile(object):
    def __init__(self, conf):
        self.root = conf.root
        self.output_dir = conf.output_dir
        self.folders_map = conf.folders_map
        self.postfix_map = conf.postfix_map
        self.data_format = conf.data_format
        # self.image_filename_re = re.compile('(.+)' + self.postfix_map['image'])

    def __call__(self, datatype, pattern=None):
        if pattern == None:
            pattern = '*%s.%s' % (self.postfix_map[datatype], self.data_format[datatype])
        search_files = os.path.join(
            self.root, self.folders_map[datatype], pattern)

        filenames = glob.glob(search_files)
        return sorted(filenames)
