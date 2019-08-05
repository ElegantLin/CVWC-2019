import mmcv
import numpy as np
from numpy import random
from PIL import Image

# import torch.nn.functional as F
from torchvision.transforms import functional as F

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


class PhotoMetricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img, boxes, labels):
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        return img, boxes, labels


class Expand(object):

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, boxes, labels):
        if random.randint(2):
            return img, boxes, labels

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        boxes += np.tile((left, top), 2)
        return img, boxes, labels


class RandomCrop(object):

    def __init__(self,
                 min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, boxes, labels):
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return img, boxes, labels

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array((int(left), int(top), int(left + new_w),
                                  int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                    center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                        center[:, 1] < patch[3])
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                return img, boxes, labels


class RandomResizeCrop(object):

    def __init__(self, scales, size):
        self.size = size
        self.scales = scales

    def __call__(self, image, bboxes, labels):
        """random resize crop
        Args:
            image: numpy array
            bboxes: [N,4]
            labels: [N, 1]
        """
        # random resize
        origin_height, origin_width, _ = image.shape
        image = Image.fromarray(image.astype(np.uint8))
        scale_ind = random.randint(0, len(self.scales)-1)
        scale = self.scales[scale_ind]
        origin_max_size = float(max(origin_width, origin_height))
        origin_min_size = float(min(origin_width, origin_height))
        max_size = scale[0]
        min_size = scale[1]

        if origin_max_size / origin_min_size * min_size > max_size:
            min_size = int(round(max_size / origin_max_size * origin_min_size))

        if origin_height > origin_width:
            img_width = min_size
            img_height = int(min_size / origin_width * origin_height)
            ratio = min_size / origin_width
        else:
            img_height = min_size
            img_width = int(min_size / origin_height * origin_width)
            ratio = min_size / origin_height

        bboxes = ratio * bboxes
        image = F.resize(image, (img_height, img_width))

        # random crop
        if img_height > img_width:
            crop_h, crop_w = self.size[0], self.size[1]
        else:
            crop_w, crop_h = self.size[0], self.size[1]

        crop_w = min(crop_w, img_width)
        crop_h = min(crop_h, img_height)

        # random select a box
        rand_index = np.random.randint(len(bboxes))
        box = bboxes[rand_index]
        # box = random.choice(bboxes)
        ctr_x = ((box[0] + box[2]) / 2.0).item()
        ctr_y = ((box[1] + box[3]) / 2.0).item()

        noise_h = random.randint(-10, 10)
        noise_w = random.randint(-30, 30)
        start_h = int(round(ctr_y - crop_h / 2)) + noise_h
        start_w = int(round(ctr_x - crop_w / 2)) + noise_w
        end_h = start_h + crop_h
        end_w = start_w + crop_w

        if start_h < 0:
            off = -start_h
            start_h += off
            end_h += off
        if start_w < 0:
            off = -start_w
            start_w += off
            end_w += off
        if end_h > img_height:
            off = end_h - img_height
            end_h -= off
            start_h -= off
        if end_w > img_width:
            off = end_w - img_width
            end_w -= off
            start_w -= off

        crop_rect = (start_w, start_h, end_w, end_h)

        box_center_x = (bboxes[:, 2] + bboxes[:, 0]) / 2.0
        box_center_y = (bboxes[:, 3] + bboxes[:, 1]) / 2.0

        mask_x = (box_center_x > crop_rect[0]) * (box_center_x < crop_rect[2])
        mask_y = (box_center_y > crop_rect[1]) * (box_center_y < crop_rect[3])
        mask = mask_x * mask_y

        bboxes = bboxes[mask]
        bboxes[:, 0] -= crop_rect[0]
        bboxes[:, 2] -= crop_rect[0]
        bboxes[:, 1] -= crop_rect[1]
        bboxes[:, 3] -= crop_rect[1]
        labels = labels[mask]

        # clip
        bboxes[:, 0] = bboxes[:, 0].clip(min=0, max=crop_w)
        bboxes[:, 1] = bboxes[:, 1].clip(min=0, max=crop_h)
        bboxes[:, 2] = bboxes[:, 2].clip(min=0, max=crop_w)
        bboxes[:, 3] = bboxes[:, 3].clip(min=0, max=crop_h)

        keep = (bboxes[:, 3] > bboxes[:, 1]) & (bboxes[:, 2] > bboxes[:, 0])
        bboxes = bboxes[keep]
        labels = labels[keep]

        image = F.crop(image, start_h, start_w, crop_h, crop_w)
        image = np.array(image).astype(np.float32)
        return image, bboxes, labels


class ExtraAugmentation(object):

    def __init__(self,
                 photo_metric_distortion=None,
                 expand=None,
                 random_crop=None,
                 rand_resize_crop=None):
        self.transforms = []
        if photo_metric_distortion is not None:
            self.transforms.append(
                PhotoMetricDistortion(**photo_metric_distortion))
        if expand is not None:
            self.transforms.append(Expand(**expand))
        if random_crop is not None:
            self.transforms.append(RandomCrop(**random_crop))
        if rand_resize_crop is not None:
            self.transforms.append(RandomResizeCrop(**rand_resize_crop))

    def __call__(self, img, boxes, labels):
        img = img.astype(np.float32)
        for transform in self.transforms:
            img, boxes, labels = transform(img, boxes, labels)
        return img, boxes, labels
