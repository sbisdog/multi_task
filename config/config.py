import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append("../")

from public.detection.dataset.cocodataset import CocoDetection, Resize, RandomFlip, RandomCrop, RandomTranslate, Normalize

import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Config(object):
    version = 1
    log = './multi_class/log_' + str(version)  # Path to save log
    checkpoint_path = '/home/jovyan/data-vol-polefs-1/multi_class/v{}'.format(version)  # Path to store checkpoint model
    resume = '/home/jovyan/data-vol-polefs-1/multi_class/v{}/latest.pth'.format(version)  # load checkpoint model
    pre_model_dir = '/home/jovyan/data-vol-polefs-1/multi_class/v{}/best.pth'.format(version)
    evaluate = None  # evaluate model path
    base_path = '/home/jovyan/data-vol-polefs-1/dataset/coco'
    train_dataset_path = os.path.join(base_path, 'train2017')
    val_dataset_path = os.path.join(base_path, 'val2017')
    dataset_annotations_path = os.path.join(base_path, 'annotations')
    
    
    
    network = "resnet50_yolofdc5"
    seed = 0
    #resize
    input_image_size = 512
    num_classes = 1

    train_dataset = CocoDetection(image_root_dir=train_dataset_path,
                                  annotation_root_dir=dataset_annotations_path,
                                  set="train2017",
                                  transform=transforms.Compose([
                                      RandomFlip(flip_prob=0.5),
                                      RandomCrop(crop_prob=0.5),
                                      RandomTranslate(translate_prob=0.5),
                                      Normalize(),
                                      Resize(resize=input_image_size)
                                  ]))
    val_dataset = CocoDetection(image_root_dir=val_dataset_path,
                                annotation_root_dir=dataset_annotations_path,
                                set="val2017",
                                transform=transforms.Compose([
                                    Normalize(),
                                    Resize(resize=input_image_size),
                                ]))
    epochs = 24
    per_node_batch_size = 4
    lr = 1e-4
    num_workers = 4
    print_interval = 100
    apex = True
    sync_bn = False

    #use the pretrained backbone
    pretrained=True
    #freeze the backbone and neck
    freeze = False
    #load the previous model to train
    previous = False
    #train multi task head
    multi_task = False


    #fpn
    #use yolof neck
    use_yolof = True
    #fpn encode channels
    fpn_out_channels=512
    use_p5=True
    
    #head
    class_num=80
    #use gn in head
    use_GN_head=False
    prior=0.01
    cnt_on_reg=True
    head_planes = 512

    #training
    #down sample strides
    strides=[16]
    #limit in decoder 
    limit_range=[[-1,512]]
    #scales parameter in head
    scales = [1.0]

    #inference
    score_threshold=0.05
    nms_iou_threshold=0.6
    max_detection_num=1000