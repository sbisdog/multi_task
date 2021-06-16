import os
import cv2
import json
import math
import numpy as np
import sys
import torch
import torch.utils.data as data
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ctn_datatools import get_border, get_affine_transform, affine_transform, color_aug
from ctn_datatools import draw_umich_gaussian, gaussian_radius

COCO_NAMES = ['__background__','Person', 'Sneakers', 'Chair', 'Other Shoes', 'Hat', 'Car', 'Lamp', 'Glasses', 'Bottle', 'Desk', 'Cup', 'Street Lights', 'Cabinet/shelf', 'Handbag/Satchel', 'Bracelet', 'Plate', 'Picture/Frame', 'Helmet', 'Book', 'Gloves', 'Storage box', 'Boat', 'Leather Shoes', 'Flower', 'Bench', 'Potted Plant', 'Bowl/Basin', 'Flag', 'Pillow', 'Boots', 'Vase', 'Microphone', 'Necklace', 'Ring', 'SUV', 'Wine Glass', 'Belt', 'Moniter/TV', 'Backpack', 'Umbrella', 'Traffic Light', 'Speaker', 'Watch', 'Tie', 'Trash bin Can', 'Slippers', 'Bicycle', 'Stool', 'Barrel/bucket', 'Van', 'Couch', 'Sandals', 'Bakset', 'Drum', 'Pen/Pencil', 'Bus', 'Wild Bird', 'High Heels', 'Motorcycle', 'Guitar', 'Carpet', 'Cell Phone', 'Bread', 'Camera', 'Canned', 'Truck', 'Traffic cone', 'Cymbal', 'Lifesaver', 'Towel', 'Stuffed Toy', 'Candle', 'Sailboat', 'Laptop', 'Awning', 'Bed', 'Faucet', 'Tent', 'Horse', 'Mirror', 'Power outlet', 'Sink', 'Apple', 'Air Conditioner', 'Knife', 'Hockey Stick', 'Paddle', 'Pickup Truck', 'Fork', 'Traffic Sign', 'Ballon', 'Tripod', 'Dog', 'Spoon', 'Clock', 'Pot', 'Cow', 'Cake', 'Dinning Table', 'Sheep', 'Hanger', 'Blackboard/Whiteboard', 'Napkin', 'Other Fish', 'Orange/Tangerine', 'Toiletry', 'Keyboard', 'Tomato', 'Lantern', 'Machinery Vehicle', 'Fan', 'Green Vegetables', 'Banana', 'Baseball Glove', 'Airplane', 'Mouse', 'Train', 'Pumpkin', 'Soccer', 'Skiboard', 'Luggage', 'Nightstand', 'Tea pot', 'Telephone', 'Trolley', 'Head Phone', 'Sports Car', 'Stop Sign', 'Dessert', 'Scooter', 'Stroller', 'Crane', 'Remote', 'Refrigerator', 'Oven', 'Lemon', 'Duck', 'Baseball Bat', 'Surveillance Camera', 'Cat', 'Jug', 'Broccoli', 'Piano', 'Pizza', 'Elephant', 'Skateboard', 'Surfboard', 'Gun', 'Skating and Skiing shoes', 'Gas stove', 'Donut', 'Bow Tie', 'Carrot', 'Toilet', 'Kite', 'Strawberry', 'Other Balls', 'Shovel', 'Pepper', 'Computer Box', 'Toilet Paper', 'Cleaning Products', 'Chopsticks', 'Microwave', 'Pigeon', 'Baseball', 'Cutting/chopping Board', 'Coffee Table', 'Side Table', 'Scissors', 'Marker', 'Pie', 'Ladder', 'Snowboard', 'Cookies', 'Radiator', 'Fire Hydrant', 'Basketball', 'Zebra', 'Grape', 'Giraffe', 'Potato', 'Sausage', 'Tricycle', 'Violin', 'Egg', 'Fire Extinguisher', 'Candy', 'Fire Truck', 'Billards', 'Converter', 'Bathtub', 'Wheelchair', 'Golf Club', 'Briefcase', 'Cucumber', 'Cigar/Cigarette ', 'Paint Brush', 'Pear', 'Heavy Truck', 'Hamburger', 'Extractor', 'Extention Cord', 'Tong', 'Tennis Racket', 'Folder', 'American Football', 'earphone', 'Mask', 'Kettle', 'Tennis', 'Ship', 'Swing', 'Coffee Machine', 'Slide', 'Carriage', 'Onion', 'Green beans', 'Projector', 'Frisbee', 'Washing Machine/Drying Machine', 'Chicken', 'Printer', 'Watermelon', 'Saxophone', 'Tissue', 'Toothbrush', 'Ice cream', 'Hotair ballon', 'Cello', 'French Fries', 'Scale', 'Trophy', 'Cabbage', 'Hot dog', 'Blender', 'Peach', 'Rice', 'Wallet/Purse', 'Volleyball', 'Deer', 'Goose', 'Tape', 'Tablet', 'Cosmetics', 'Trumpet', 'Pineapple', 'Golf Ball', 'Ambulance', 'Parking meter', 'Mango', 'Key', 'Hurdle', 'Fishing Rod', 'Medal', 'Flute', 'Brush', 'Penguin', 'Megaphone', 'Corn', 'Lettuce', 'Garlic', 'Swan', 'Helicopter', 'Green Onion', 'Sandwich', 'Nuts', 'Speed Limit Sign', 'Induction Cooker', 'Broom', 'Trombone', 'Plum', 'Rickshaw', 'Goldfish', 'Kiwi fruit', 'Router/modem', 'Poker Card', 'Toaster', 'Shrimp', 'Sushi', 'Cheese', 'Notepaper', 'Cherry', 'Pliers', 'CD', 'Pasta', 'Hammer', 'Cue', 'Avocado', 'Hamimelon', 'Flask', 'Mushroon', 'Screwdriver', 'Soap', 'Recorder', 'Bear', 'Eggplant', 'Board Eraser', 'Coconut', 'Tape Measur/ Ruler', 'Pig', 'Showerhead', 'Globe', 'Chips', 'Steak', 'Crosswalk Sign', 'Stapler', 'Campel', 'Formula 1 ', 'Pomegranate', 'Dishwasher', 'Crab', 'Hoverboard', 'Meat ball', 'Rice Cooker', 'Tuba', 'Calculator', 'Papaya', 'Antelope', 'Parrot', 'Seal', 'Buttefly', 'Dumbbell', 'Donkey', 'Lion', 'Urinal', 'Dolphin', 'Electric Drill', 'Hair Dryer', 'Egg tart', 'Jellyfish', 'Treadmill', 'Lighter', 'Grapefruit', 'Game board', 'Mop', 'Radish', 'Baozi', 'Target', 'French', 'Spring Rolls', 'Monkey', 'Rabbit', 'Pencil Case', 'Yak', 'Red Cabbage', 'Binoculars', 'Asparagus', 'Barbell', 'Scallop', 'Noddles', 'Comb', 'Dumpling', 'Oyster', 'Table Teniis paddle', 'Cosmetics Brush/Eyeliner Pencil', 'Chainsaw', 'Eraser', 'Lobster', 'Durian', 'Okra', 'Lipstick', 'Cosmetics Mirror', 'Curling', 'Table Tennis ']

COCO_IDS = [i for i in range(1, 366)]

COCO_MEAN = [0.40789654, 0.44719302, 0.47026115]
COCO_STD = [0.28863828, 0.27408164, 0.27809835]
COCO_EIGEN_VALUES = [0.2141788, 0.01817699, 0.00341571]
COCO_EIGEN_VECTORS = [[-0.58752847, -0.69563484, 0.41340352],
                      [-0.5832747, 0.00994535, -0.81221408],
                      [-0.56089297, 0.71832671, 0.41158938]]


class COCO(data.Dataset):
    def __init__(self, data_dir, split, split_ratio=1.0, img_size=512):
        super(COCO, self).__init__()
        self.num_classes = 365
        self.class_name = COCO_NAMES
        self.valid_ids = COCO_IDS
        self.cat_ids = {v: i for i, v in enumerate(self.valid_ids)}

        self.data_rng = np.random.RandomState(123)
        self.eig_val = np.array(COCO_EIGEN_VALUES, dtype=np.float32)
        self.eig_vec = np.array(COCO_EIGEN_VECTORS, dtype=np.float32)
        self.mean = np.array(COCO_MEAN, dtype=np.float32)[None, None, :]
        self.std = np.array(COCO_STD, dtype=np.float32)[None, None, :]

        self.split = split
        self.data_dir = os.path.join(data_dir, 'coco')
        self.img_dir = os.path.join(self.data_dir, '%s2017' % split)
        if split == 'test':
            self.annot_path = os.path.join(
                self.data_dir, 'annotations', 'image_info_test-dev2017.json')
        else:
            self.annot_path = os.path.join(
                self.data_dir, 'annotations', 'instances_%s2017.json' % split)

        self.max_objs = 700
        self.padding = 127  # 31 for resnet/resdcn
        self.down_ratio = 4
        self.img_size = {'h': img_size, 'w': img_size}
        self.fmap_size = {'h': img_size // self.down_ratio,
                          'w': img_size // self.down_ratio}
        self.rand_scales = np.arange(0.6, 1.4, 0.1)
        self.gaussian_iou = 0.7

        print('==> initializing coco 2017 %s data.' % split)
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()

        if 0 < split_ratio < 1:
            split_size = int(
                np.clip(split_ratio * len(self.images), 1, len(self.images)))
            self.images = self.images[:split_size]

        self.num_samples = len(self.images)

        print('Loaded %d %s samples' % (self.num_samples, split))

    def __getitem__(self, index):
        img_id = self.images[index]
        img_path = os.path.join(
            self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        annotations = self.coco.loadAnns(ids=ann_ids)
        labels = np.array([self.cat_ids[anno['category_id']]
                          for anno in annotations])
        bboxes = np.array([anno['bbox']
                          for anno in annotations], dtype=np.float32)
        if len(bboxes) == 0:
            bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
            labels = np.array([[0]])
        bboxes[:, 2:] += bboxes[:, :2]  # xywh to xyxy

        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        center = np.array([width / 2., height / 2.],
                          dtype=np.float32)  # center of image
        scale = max(height, width) * 1.0

        flipped = False
        if self.split == 'train':
            scale = scale * np.random.choice(self.rand_scales)
            w_border = get_border(128, width)
            h_border = get_border(128, height)
            center[0] = np.random.randint(low=w_border, high=width - w_border)
            center[1] = np.random.randint(low=h_border, high=height - h_border)

            if np.random.random() < 0.5:
                flipped = True
                img = img[:, ::-1, :]
                center[0] = width - center[0] - 1

        trans_img = get_affine_transform(
            center, scale, 0, [self.img_size['w'], self.img_size['h']])
        img = cv2.warpAffine(
            img, trans_img, (self.img_size['w'], self.img_size['h']))

        # -----------------------------------debug---------------------------------
        # for bbox, label in zip(bboxes, labels):
        #   if flipped:
        #     bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        #   bbox[:2] = affine_transform(bbox[:2], trans_img)
        #   bbox[2:] = affine_transform(bbox[2:], trans_img)
        #   bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.img_size['w'] - 1)
        #   bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.img_size['h'] - 1)
        #   cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        #   cv2.putText(img, self.class_name[label + 1], (int(bbox[0]), int(bbox[1])),
        #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # cv2.imshow('img', img)
        # cv2.waitKey()
        # -----------------------------------debug---------------------------------

        img = img.astype(np.float32) / 255.

        if self.split == 'train':
            color_aug(self.data_rng, img, self.eig_val, self.eig_vec)

        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)  # from [H, W, C] to [C, H, W]

        trans_fmap = get_affine_transform(
            center, scale, 0, [self.fmap_size['w'], self.fmap_size['h']])

        hmap = np.zeros(
            (self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)  # heatmap
        # width and height
        w_h_ = np.zeros((self.max_objs, 2), dtype=np.float32)
        regs = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression
        inds = np.zeros((self.max_objs,), dtype=np.int64)
        ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)

        # detections = []
        for k, (bbox, label) in enumerate(zip(bboxes, labels)):
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_fmap)
            bbox[2:] = affine_transform(bbox[2:], trans_fmap)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.fmap_size['w'] - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.fmap_size['h'] - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                obj_c = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                obj_c_int = obj_c.astype(np.int32)

                radius = max(0, int(gaussian_radius(
                    (math.ceil(h), math.ceil(w)), self.gaussian_iou)))
                draw_umich_gaussian(hmap[label], obj_c_int, radius)
                w_h_[k] = 1. * w, 1. * h
                regs[k] = obj_c - obj_c_int  # discretization error
                inds[k] = obj_c_int[1] * self.fmap_size['w'] + obj_c_int[0]
                ind_masks[k] = 1
                # groundtruth bounding box coordinate with class
                # detections.append([obj_c[0] - w / 2, obj_c[1] - h / 2,
                #                    obj_c[0] + w / 2, obj_c[1] + h / 2, 1, label])

        # detections = np.array(detections, dtype=np.float32) \
        #   if len(detections) > 0 else np.zeros((1, 6), dtype=np.float32)

        return {'image': img,
                'hmap': hmap, 'w_h_': w_h_, 'regs': regs, 'inds': inds, 'ind_masks': ind_masks,
                'c': center, 's': scale, 'img_id': img_id}

    def __len__(self):
        return self.num_samples


class COCO_eval(COCO):
    def __init__(self, data_dir, split, test_scales=(1,), test_flip=False, fix_size=False):
        super(COCO_eval, self).__init__(data_dir, split)
        self.test_flip = test_flip
        self.test_scales = test_scales
        self.fix_size = fix_size

    def __getitem__(self, index):
        img_id = self.images[index]
        img_path = os.path.join(
            self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])
        image = cv2.imread(img_path)
        height, width = image.shape[0:2]

        out = {}
        for scale in self.test_scales:
            new_height = int(height * scale)
            new_width = int(width * scale)

            if self.fix_size:
                img_height, img_width = self.img_size['h'], self.img_size['w']
                center = np.array(
                    [new_width / 2., new_height / 2.], dtype=np.float32)
                scaled_size = max(height, width) * 1.0
                scaled_size = np.array(
                    [scaled_size, scaled_size], dtype=np.float32)
            else:
                img_height = (new_height | self.padding) + 1
                img_width = (new_width | self.padding) + 1
                center = np.array(
                    [new_width // 2, new_height // 2], dtype=np.float32)
                scaled_size = np.array(
                    [img_width, img_height], dtype=np.float32)

            img = cv2.resize(image, (new_width, new_height))
            trans_img = get_affine_transform(
                center, scaled_size, 0, [img_width, img_height])
            img = cv2.warpAffine(img, trans_img, (img_width, img_height))

            img = img.astype(np.float32) / 255.
            img -= self.mean
            img /= self.std
            # from [H, W, C] to [1, C, H, W]
            img = img.transpose(2, 0, 1)[None, :, :, :]

            if self.test_flip:
                img = np.concatenate((img, img[:, :, :, ::-1].copy()), axis=0)

            out[scale] = {'image': img,
                          'center': center,
                          'scale': scaled_size,
                          'fmap_h': img_height // self.down_ratio,
                          'fmap_w': img_width // self.down_ratio}

        return img_id, out

    def convert_eval_format(self, all_bboxes):
        # all_bboxes: num_samples x num_classes x 5
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self.valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(
                        map(lambda x: float("{:.2f}".format(x)), bbox[0:4]))

                    detection = {"image_id": int(image_id),
                                 "category_id": int(category_id),
                                 "bbox": bbox_out,
                                 "score": float("{:.2f}".format(score))}
                    detections.append(detection)
        return detections

    def run_eval(self, results, save_dir=None):
        detections = self.convert_eval_format(results)

        if save_dir is not None:
            result_json = os.path.join(save_dir, "results.json")
            json.dump(detections, open(result_json, "w"))

        coco_dets = self.coco.loadRes(detections)
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats

    @staticmethod
    def collate_fn(batch):
        out = []
        for img_id, sample in batch:
            out.append((img_id, {s: {k: torch.from_numpy(sample[s][k]).float()
                                     if k == 'image' else np.array(sample[s][k]) for k in sample[s]} for s in sample}))
        return out


if __name__ == '__main__':
    from tqdm import tqdm
    import pickle

    dataset = COCO('/home/jovyan/data-vol-1/zhangze/code/CenterNet-better/datasets', 'train')
#     for d in dataset:
#         b1 = d
#     #   pass

#     pass
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                               shuffle=False, num_workers=16,
                                               pin_memory=True, drop_last=True)
    #
    for b in tqdm(train_loader):
        a = b
        pass
