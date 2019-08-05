from .baseline import Baseline
from .pcb_model import PCBModel
from .sync_bn import convert_model

def build_model(cfg, num_classes):
    if cfg.MODEL.NAME == 'resnet50':
        model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE,
                cfg.MODEL.PRETRAIN_PATH)
    elif cfg.MODEL.NAME == 'pcb_model':
        model = PCBModel(num_classes, cfg.MODEL.LAST_STRIDE,
                cfg.MODEL.PRETRAIN_PATH)
    return model

