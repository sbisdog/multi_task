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
    version = 2
    log = './multi_task/log_' + str(version)  # Path to save log
    checkpoint_path = '/home/jovyan/data-vol-polefs-1/small_sample/multi_task/checkpoints/v{}'.format(version)  # Path to store checkpoint model
    resume = '/home/jovyan/data-vol-polefs-1/small_sample/multi_task/checkpoints/v{}/latest.pth'.format(version)  # load checkpoint model
    pre_model_dir = '/home/jovyan/data-vol-polefs-1/small_sample/multi_task/checkpoints/v1/best.pth'.format(version-1)
    evaluate = None  # evaluate model path
    base_path = '/home/jovyan/data-vol-polefs-1/small_sample/dataset'
    train_dataset_path = os.path.join(base_path, 'images/images')
    val_dataset_path = os.path.join(base_path, 'images/images')
    dataset_annotations_path = os.path.join("/home/jovyan/data-vol-polefs-1/small_sample/multi_task", 'annotations/v{}'.format(version))
    
    
    
    network = "resnet50_yolof"
    seed = 0
    #resize
    input_image_size = 667
    num_classes = 1

    train_dataset = CocoDetection(image_root_dir=train_dataset_path,
                                  annotation_root_dir=dataset_annotations_path,
                                  set="train",
                                  transform=transforms.Compose([
                                      RandomFlip(flip_prob=0.5),
                                      RandomCrop(crop_prob=0.5),
                                      RandomTranslate(translate_prob=0.5),
                                      Normalize(),
                                      Resize(resize=input_image_size)
                                  ]))
    val_dataset = CocoDetection(image_root_dir=val_dataset_path,
                                annotation_root_dir=dataset_annotations_path,
                                set="val",
                                transform=transforms.Compose([
                                    Normalize(),
                                    Resize(resize=input_image_size),
                                ]))
    
#***********************************************************#
    #use the pretrained backbone
    pretrained=True
    #freeze the backbone and neck
#     freeze = False
    #load the previous model to train,if v1:use the coco_pretrained,else use the latest version model
#     previous = False
    #train multi task head
    multi_task = False

#***********************************************************#
    #fpn
    #use yolof neck
    use_yolof = True
    #fpn encode channels
    fpn_out_channels=512
    use_p5=True
    
#***********************************************************#    
    #head
    class_num=1
    #use gn in head
    use_GN_head=False
    prior=0.01
    cnt_on_reg=True
    #yolof
    yolof_encoder_channels = 512
    #ttf up
    use_ttf = True
    ttf_out_channels = [256, 128]
    
#***********************************************************#
    #training
    epochs = 24
    per_node_batch_size = 4
    lr = 1e-4
    num_workers = 4
    print_interval = 100
    eval_interval = 4
    apex = True
    sync_bn = False
    #down sample strides
    strides=[16]
    #limit in decoder 
#     limit_range=[[-1,99999]]
    limit_range=[[-1,667]]
    #scales parameter in head
    scales = [1.0]

#***********************************************************#
    #inference
    score_threshold=0.05
    nms_iou_threshold=0.6
    max_detection_num=1000
