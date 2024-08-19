from .builder import DATASETS
from .custom import CustomDataset, CustomDatasetWithEleWiseWeights


@DATASETS.register_module()
class AppleFarmSimDataset(CustomDataset):
    """ Apple Farm Segmentation Dataset (Sim)

    In segmentation map annotations, 255 stands for background, which is ignored.

    Images have suffix ".jpg" while annotations have ".png".
    """
    CLASSES = ("sky", "terrain", "low-vegetation", "trunk", "canopy", "building", "others")
    PALETTE = ([60, 112, 164], [152, 251, 152], [147, 217, 163], [194, 118, 100], [97, 135, 110], [244, 242, 222], [250, 170, 30])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(img_suffix=img_suffix,
                         seg_map_suffix=seg_map_suffix,
                         reduce_zero_label=reduce_zero_label,
                         **kwargs)


@DATASETS.register_module()
class AppleFarmRealDataset(CustomDatasetWithEleWiseWeights):
    """ Apple Farm Segmentation Dataset (Real) with optional element-wise weights

    In segmentation map annotations, 255 stands for background, which is ignored.

    Images have suffix ".JPG" while annotations have ".png".
    """
    CLASSES = ("sky", "terrain", "low-vegetation", "trunk", "canopy", "building", "others")
    PALETTE = ([60, 112, 164], [152, 251, 152], [147, 217, 163], [194, 118, 100], [97, 135, 110], [244, 242, 222], [250, 170, 30])

    def __init__(self,
                 img_suffix='.JPG',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(img_suffix=img_suffix,
                         seg_map_suffix=seg_map_suffix,
                         reduce_zero_label=reduce_zero_label,
                         **kwargs)
