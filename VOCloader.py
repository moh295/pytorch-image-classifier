import torch
import collections
import os
from xml.etree.ElementTree import Element as ET_Element

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import  verify_str_arg

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, List

from PIL import Image

from torch.utils.data import DataLoader
from torchvision import  transforms

#
#
# DATASET_YEAR_DICT = {
#     "2012": {
#         "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
#         "filename": "VOCtrainval_11-May-2012.tar",
#         "md5": "6cd6e144f989b92b3379bac3b3de84fd",
#         "base_dir": os.path.join("VOCdevkit", "VOC2012"),
#     },
#     "2011": {
#         "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar",
#         "filename": "VOCtrainval_25-May-2011.tar",
#         "md5": "6c3384ef61512963050cb5d687e5bf1e",
#         "base_dir": os.path.join("TrainVal", "VOCdevkit", "VOC2011"),
#     },
#     "2010": {
#         "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar",
#         "filename": "VOCtrainval_03-May-2010.tar",
#         "md5": "da459979d0c395079b5c75ee67908abb",
#         "base_dir": os.path.join("VOCdevkit", "VOC2010"),
#     },
#     "2009": {
#         "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar",
#         "filename": "VOCtrainval_11-May-2009.tar",
#         "md5": "a3e00b113cfcfebf17e343f59da3caa1",
#         "base_dir": os.path.join("VOCdevkit", "VOC2009"),
#     },
#     "2008": {
#         "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar",
#         "filename": "VOCtrainval_11-May-2012.tar",
#         "md5": "2629fa636546599198acfcfbfcf1904a",
#         "base_dir": os.path.join("VOCdevkit", "VOC2008"),
#     },
#     "2007": {
#         "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
#         "filename": "VOCtrainval_06-Nov-2007.tar",
#         "md5": "c52e279531787c972589f7e41ab4ae64",
#         "base_dir": os.path.join("VOCdevkit", "VOC2007"),
#     },
#     "2007-test": {
#         "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
#         "filename": "VOCtest_06-Nov-2007.tar",
#         "md5": "b6e924de25625d8de591ea690078ad9f",
#         "base_dir": os.path.join("VOCdevkit", "VOC2007"),
#     },
# }
#

class _VOCBase(VisionDataset):
    _SPLITS_DIR: str
    _TARGET_DIR: str
    _TARGET_FILE_EXT: str

    def __init__(
        self,
        root: str,
        # year: str = "2007",
        image_set: str = "train",
        transform: Optional[Callable] = None,
        # target_transform: Optional[Callable] = None,
        # transforms: Optional[Callable] = None,
    ):
        # super().__init__(root, transforms, transform, target_transform)
        super().__init__(root,transform)
        # if year == "2007-test":
        #     if image_set == "test":
        #         warnings.warn(
        #             "Accessing the test image set of the year 2007 with year='2007-test' is deprecated "
        #             "since 0.12 and will be removed in 0.14. "
        #             "Please use the combination year='2007' and image_set='test' instead."
        #         )
        #         year = "2007"
        #     else:
        #         raise ValueError(
        #             "In the test image set of the year 2007 only image_set='test' is allowed. "
        #             "For all other image sets use year='2007' instead."
        #         )
        # self.year = year

        valid_image_sets = ["train", "trainval", "val"]
        # if year == "2007":
        #     valid_image_sets.append("test")
        self.image_set = verify_str_arg(image_set, "image_set", valid_image_sets)

        # key = "2007-test" if year == "2007" and image_set == "test" else year
        # dataset_year_dict = DATASET_YEAR_DICT[key]

        # self.url = dataset_year_dict["url"]
        # self.filename = dataset_year_dict["filename"]
        # self.md5 = dataset_year_dict["md5"]
        #
        # base_dir = dataset_year_dict["base_dir"]
        # voc_root = os.path.join(self.root, base_dir)

        voc_root=self.root
        if not os.path.isdir(voc_root):
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR)
        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
        with open(os.path.join(split_f)) as f:
            file_names = [x.strip() for x in f.readlines()]

        image_dir = os.path.join(voc_root, "JPEGImages")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        target_dir = os.path.join(voc_root, self._TARGET_DIR)
        self.targets = [os.path.join(target_dir, x + self._TARGET_FILE_EXT) for x in file_names]

        assert len(self.images) == len(self.targets)

    def __len__(self) -> int:
        return len(self.images)



class VOCSegmentation(_VOCBase):


    _SPLITS_DIR = "Segmentation"
    _TARGET_DIR = "SegmentationClass"
    _TARGET_FILE_EXT = ".png"

    @property
    def masks(self) -> List[str]:
        return self.targets


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target




class VOCDetection(_VOCBase):

    _SPLITS_DIR = "Main"
    _TARGET_DIR = "Annotations"
    _TARGET_FILE_EXT = ".xml"

    @property
    def annotations(self) -> List[str]:
        return self.targets


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target_dict = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())
        boxes=[]
        labels=[]
        labels_dict=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
        for lb in target_dict['annotation']['object']:
            print('lb', lb)
            #  print('hand side', lb['handside'])
            # print(lb['name'])

            # print('label',lb['name'])
            id=0
            for name in labels_dict:
                if name==lb['name']:
                    labels.append(id)
                    break
                else: id+=1



            # if obj == 'hand':
            #     print('hand side', lb['handside'])
            # print('labels', lb['bndbox'])
            box = []
            box.append(int(lb['bndbox']['xmin']))
            box.append(int(lb['bndbox']['ymin']))
            box.append(int(lb['bndbox']['xmax']))
            box.append(int(lb['bndbox']['ymax']))
            boxes.append(box)


        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor((labels), dtype=torch.int64)


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


    @staticmethod
    def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(VOCDetection.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict



def dataloader(batch_size=1,input_size=300):
    transform = transforms.Compose(
        [
            # transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = VOCDetection(root='./data/VOCdevkit/VOC2007', image_set='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2)


    val_dataset =VOCDetection(root='./data/VOCdevkit/VOC2007', image_set='val',transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    trainval_dataset = VOCDetection(root='./data/VOCdevkit/VOC2007', image_set='trainval', transform=transform)
    trainval_loader = DataLoader(trainval_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    print('val_dataset',val_dataset[0])


    return train_loader,trainval_loader ,val_loader
    # val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=True, num_workers=2)
    # train_dataset = VOCDetection(root='./data', year='2007', image_set='train',transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size,
    # shuffle=True, num_workers=2)
