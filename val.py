# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import cv2
import numpy as np
import json
from pycocotools.coco import COCO
import os
from copy import deepcopy
from public.detection.models import fcos, centernet, centernet_multi
from public.detection.models.decode import FCOSDecoder,CenterNetDecoder
from tqdm import tqdm
import random
import sys
import time
from PIL import Image,ImageFont,ImageDraw

label_map = ["三字经劲书", "1000步的缤纷台湾", "Hello你好", "产业劲报", "厨娘香Q秀", "大陆新闻解读", "大千世界", "地球旅馆", "好样Formosa","胡乃文开讲", "环球直击", "健康1+1", "健谈交流", "杰森视角", "今日点击", "今日加州", "经典艺术", "九评共产党", "美国思想领袖", "美味亚洲","热点互动", "社区广角镜", "谈古论今话中医", "探索时分", "天庭小子小乾坤", "我的音乐想想", "我们告诉未来", "午间新闻", "笑谈风云", "新唐人全球新闻", "新唐人周刊", "新闻看点","新闻大家谈", "新闻拍案惊奇", "馨香雅句", "亚太新闻", "一周经济回顾", "远见快评", "早安新唐人", "珍言珍语", "中国禁闻"]


use_gpu = True
if_use_json = True
if_centernet = True
if_multi = True
resize = 512
#score阈值
threshold = 0.2

#输入图片地址
# im_dir = '/home/jovyan/data-vol-1/zhangze/code/multi_task/test_imgs'
im_dir = os.path.join('/home/jovyan/data-vol-polefs-1/dataset/taibiao', 'images/val2017')
#输出图片地址
out_dir = './out_imgs_task2/'
#val json 地址
json_dir = '/home/jovyan/data-vol-polefs-1/dataset/taibiao/annotations/instances_val2017.json'

if if_centernet:
    if if_multi:
        model_dir = '/home/jovyan/data-vol-1/zhangze/code/multi_task/train/checkpoints_multi_v5/best.pth'
    else:
#         model_dir = '/home/jovyan/data-vol-1/zhangze/code/multi_task/train/checkpoints_multi_v2/best.pth'
        model_dir = '/home/jovyan/data-vol-1/zhangze/code/pytorch-ImageNet-CIFAR-COCO-VOC-training/detection_experiments/resnet18_centernet_coco_distributed_apex_resize512/checkpoints/best.pth'
else:
    model_dir = '/home/jovyan/data-vol-polefs-1/res18_crops_checkpoints_v5_boso30_dirty/best.pth'

#清空输出文件夹
os.system("rm -f {}*".format(out_dir))

if if_use_json:
    im_dir = os.path.join('/home/jovyan/data-vol-polefs-1/dataset/taibiao', 'images/val2017')
    coco = COCO(os.path.join('/home/jovyan/data-vol-polefs-1/dataset/taibiao', 'annotations/instances_val2017.json'))
    im_ids = coco.getImgIds(catIds=1)
    anns_ids = {coco.loadImgs(ids=item)[0]["file_name"]:coco.getAnnIds(imgIds=item)[0] for item in im_ids}

    im_list = [item["file_name"] for item in coco.loadImgs(ids=coco.getImgIds(catIds=1))]
else:
    im_list = os.listdir(im_dir)

#加载模型 
torch.cuda.set_device(0)

if if_centernet:
    if if_multi:
        model = centernet_multi.__dict__['resnet50_centernet'](**{
                "pretrained": False,
                "num_classes": [41,1],
                "multi_head": True,
                "selayer": True,
                "use_ttf": False
            })
        decoder = CenterNetDecoder(image_w=512,
                                   image_h=512)
    else:
        model = centernet_multi.__dict__['resnet50_centernet'](**{
                "pretrained": False,
                "num_classes": [1]
            })
        decoder = CenterNetDecoder(image_w=512,
                                   image_h=512)
else:
    model = fcos.__dict__['resnet18_fcos'](**{
            "pretrained": False,
            "num_classes": 41,
            "use_TransConv": True,
            "use_gn": False,
            "fpn_bn":True
        })
    decoder = FCOSDecoder(image_w=resize, image_h=resize)

pre_model = torch.load(model_dir, map_location=torch.device('cpu'))


if use_gpu:
    decoder = decoder.cuda()
    model = model.cuda()

model.load_state_dict(pre_model, strict=False)
model.eval()


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


#创建输出的json dict
out = {}

count=0
num_bbox = 0
with torch.no_grad():
    for item in tqdm(im_list):
        bbox_gt = coco.loadAnns(ids=(anns_ids[item]))[0]["bbox"]
        
        if item == '.ipynb_checkpoints':
            continue
        current_dir = os.path.join(im_dir, item)
        
        img = cv2.imdecode(np.fromfile(current_dir, dtype=np.uint8), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        height, width, _ = img.shape
        max_image_size = max(height, width)
        resize_factor = resize / max_image_size
        resize_height, resize_width = int(height * resize_factor), int(
            width * resize_factor)
        img = cv2.resize(img, (resize_width, resize_height))
        resized_img = np.zeros((resize, resize, 3))
        resized_img[0:resize_height, 0:resize_width] = img
        resized_img = torch.tensor(resized_img).permute(2, 0, 1).float().unsqueeze(0)
        resized_img = resized_img.cuda()
        
        if not if_centernet:
            cls_heads, reg_heads, center_heads, batch_positions = model(resized_img)
            scores, classes, boxes = decoder(cls_heads, reg_heads, center_heads, batch_positions)
        else:
            heatmap_output, offset_output, wh_output = model(resized_img)
            scores, classes, boxes = decoder(heatmap_output, offset_output,
                                             wh_output)
        
        
        scores, classes, boxes = scores.cpu(), classes.cpu(), boxes.cpu()
        scores = scores[0].tolist()
        bboxes = (boxes[0]/resize_factor)
                
        #读取图片准备画框
#         font = ImageFont.truetype('han.ttc', 15)
#         or_im = cv2.imdecode(np.fromfile(current_dir, dtype=np.uint8), -1)
#         pil_image = Image.fromarray(cv2.cvtColor(or_im, cv2.COLOR_BGR2RGB))
#         draw = ImageDraw.Draw(pil_image)
        
        out[item] = []
        for i, score in enumerate(scores):
            bbox = bboxes[i].tolist()
            
            if score > threshold:
                num_bbox += 1
                dic = {}
                dic["scores"] = score
                dic["class"] = label_map[int(classes[0][0].item())]
                dic["bbox"] = bbox
                if compute_iou(bbox_gt, bbox) > 0.5:
                    count += 1
                    break
#                 pos = (int(bbox[0]), int(bbox[3]))
#                 text = label_map[int(classes[0][0].item())]+" " + str(score)[:4]
#                 color = (255, 0, 0)
#                 draw.text(pos,text,font=font,fill=color)
                
#                 draw.rectangle([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])], outline=(255, 255, 0), width=2)
                
#         cv_img = cv2.cvtColor(np.asarray(pil_image),cv2.COLOR_RGB2BGR)
#         cv2.imwrite(os.path.join(out_dir, item),cv_img)
print("recall: ", count/len(im_list))
print("acc : ", count/num_bbox)
json.dump(out, open(os.path.join(out_dir, "out.json"), "w"))


