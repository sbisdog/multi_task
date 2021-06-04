import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

# from public.path import COCO2017_path
from public.detection.dataset.cocodataset import CocoDetection, Resize, RandomFlip, RandomCrop, RandomTranslate, Normalize

import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Config(object):
    version = 10
    log = './log_ctn_multi_v' + str(version)  # Path to save log
    checkpoint_path = './checkpoints_multi_v' + str(version)  # Path to store checkpoint model
    resume = checkpoint_path + '/latest.pth'  # load checkpoint model
    evaluate = None  # evaluate model path
    train_dataset_path = os.path.join('/home/jovyan/data-vol-polefs-1/dataset/taibiao', 'images/train2017')
    val_dataset_path = os.path.join('/home/jovyan/data-vol-polefs-1/dataset/taibiao', 'images/val2017')
    dataset_annotations_path = os.path.join('/home/jovyan/data-vol-polefs-1/dataset/taibiao', 'annotations')
#     train_dataset_path = os.path.join('/home/jovyan/data-vol-polefs-1/dataset/lanmu', 'images/train2017')
#     val_dataset_path = os.path.join('/home/jovyan/data-vol-polefs-1/dataset/lanmu', 'images/val2017')
#     dataset_annotations_path = os.path.join('/home/jovyan/data-vol-polefs-1/dataset/lanmu', 'annotations')
#     train_dataset_path = os.path.join('/home/jovyan/data-vol-polefs-2/datasets_shenhe/scripts/03-08', 'crop_images')
#     val_dataset_path = os.path.join('/home/jovyan/data-vol-polefs-2/datasets_shenhe/scripts/03-08', 'train2017')
#     dataset_annotations_path = os.path.join('/home/jovyan/data-vol-polefs-2/datasets_shenhe/scripts/03-08', 'annotations')

    network = "resnet50_centernet"
    pretrained = False
    #*********************************************************************************
    multi_head = True
    
    #must use at the same time
    #use mlp layer after head
    cls_mlp = False
    #load the params to head2
    load_head = False
    
    #use selayer before head
    selayer = True
    #use ttf head in centernet head
    use_ttf = True
#     pre_model_dir = '/home/jovyan/data-vol-1/zhangze/code/pytorch-ImageNet-CIFAR-COCO-VOC-training/detection_experiments/resnet18_centernet_coco_distributed_apex_resize512/checkpoints/best.pth'
    pre_model_dir = "/home/jovyan/data-vol-1/zhangze/code/multi_task/train/checkpoints_multi_v2/best.pth"
    #head1 head2 classes
    num_classes = [41, 1]
    #*********************************************************************************
    seed = 0
    input_image_size = 512

    use_multi_scale = False
    multi_scale_range = [0.6, 1.4]
    stride = 4

    train_dataset = CocoDetection(image_root_dir=train_dataset_path,
                                  annotation_root_dir=dataset_annotations_path,
                                  set="train2017",
                                  transform=transforms.Compose([
                                      RandomFlip(flip_prob=0.5),
                                      RandomCrop(crop_prob=0.5),
                                      RandomTranslate(translate_prob=0.5),
                                      Normalize(),
                                  ]))

    val_dataset = CocoDetection(image_root_dir=val_dataset_path,
                                annotation_root_dir=dataset_annotations_path,
                                set="val2017",
                                transform=transforms.Compose([
                                    Normalize(),
                                    Resize(resize=input_image_size),
                                ]))

    epochs = 140
    milestones = [90, 120]
    per_node_batch_size = 16
    lr = 5e-4
#     lr = 1e-5
    num_workers = 4
    print_interval = 100
    apex = True
    sync_bn = False