import sys
import os
import argparse
import random
import shutil
import time
import warnings
import json
from collections import OrderedDict

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append("../")
warnings.filterwarnings('ignore')

import numpy as np
from thop import profile
from thop import clever_format
from tqdm import tqdm
import apex
from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from config.config_test import Config
from public.detection.dataset.cocodataset import COCODataPrefetcher, Collater
from public.detection.models.loss import FCOSLoss
from public.detection.models.decode import FCOSDecoder,CenterNetDecoder
from public.detection.models import yolof2_multiHead as yolof
from public.detection.models import centernet_multi as centernet

from public.imagenet.utils import get_logger
from pycocotools.cocoeval import COCOeval


use_ctn = True
resize = 512
model_dir = "/home/jovyan/data-vol-polefs-1/small_sample/multi_task/checkpoints/v1/best.pth"

# pre_model = torch.load(model_dir, map_location='cpu')

if not use_ctn:
    model = yolof.__dict__[Config.network](**{
            "pretrained": Config.pretrained,
            "config": Config
        })
else:
    model = centernet.__dict__['resnet50_centernet'](**{
            "pretrained": False,
            "num_classes": [1,100],
            "multi_head": True
        })
    decoder = CenterNetDecoder(image_w=512,
                               image_h=512)
# model.load_state_dict(pre_model, strict=False)


model = model.cuda().eval()
decoder = decoder.cuda()

start_time0 = time.time()
all_backbone_time = 0
all_up_time = 0
all_head1_time = 0
all_head2_time = 0
all_decoder1_time = 0
all_decoder2_time = 0
all_time = 0
if not use_ctn:
    with torch.no_grad():
        for i in range(1000):
            a = torch.randn(1,3,667,667).cuda()
            start_time = time.time()
            out = model.backbone(a)
            backbone_time = time.time() -start_time
            all_backbone_time += backbone_time

            out1 = model.clsregcnt_head_1(out)
            head1_time = time.time() - start_time - backbone_time
            all_head1_time += head1_time

    #         out2 = model.clsregcnt_head_2(out)
    #         head2_time = time.time() - start_time - backbone_time - head1_time
    #         all_head2_time += head2_time
    #         del out, out1, out2, a
            del out, out1, a

    all_time = all_backbone_time + all_head1_time + all_head2_time + all_decode_time
else:
    
    with torch.no_grad():
        for i in range(1000):
            a = torch.randn(1,3,resize,resize).cuda()
            start_time = time.time()
            out = model.backbone(a)
            backbone_time = time.time() -start_time
            all_backbone_time += backbone_time

            out1 = model.centernet_head(out[-1])
            head1_time = time.time() - start_time - backbone_time
            all_head1_time += head1_time
            
            scores, classes, boxes = decoder(out1[0], out1[1],
                                             out1[2])
            decoder1_time = time.time() - start_time - backbone_time - head1_time
            all_decoder1_time += decoder1_time

            out2 = model.centernet_head_2(out[-1])
            head2_time = time.time() - start_time - backbone_time - head1_time - decoder1_time
            all_head2_time += head2_time
            
            scores, classes, boxes = decoder(out2[0], out2[1],
                                             out2[2])
            decoder2_time = time.time() - start_time - backbone_time - head1_time - decoder1_time - head2_time
            all_decoder2_time += decoder2_time
            
            del out, out1, out2, a
#             del out, out1, a

    all_time = all_backbone_time + all_head1_time + all_head2_time + all_decoder1_time + all_decoder2_time
    

print("backbone_time: ", all_backbone_time, "ms")
print("head1_time: ", all_head1_time, "ms")
print("head2_time: ", all_head2_time, "ms")
print("all_decoder1_time: ", all_decoder1_time, "ms")
print("all_decoder2_time: ", all_decoder2_time, "ms")
print("all_time: ", all_time, "ms")

