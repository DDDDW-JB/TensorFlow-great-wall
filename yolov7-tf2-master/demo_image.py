import colorsys
import os
import time
import cv2
import gc

import numpy as np
import tensorflow as tf
from PIL import ImageDraw, ImageFont, Image
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from nets.yolo import yolo_body, fusion_rep_vgg
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox, DecodeBoxNP


class YOLO(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        "model_path": 'logs/best_epoch_weights.h5',
        "classes_path": 'model_data/cls_classes.txt',
        # ---------------------------------------------------------------------#
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        # ---------------------------------------------------------------------#
        "anchors_path": 'model_data/yolo_anchors.txt',
        "anchors_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        # ---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        # ---------------------------------------------------------------------#
        "input_shape": [640, 640],
        # ---------------------------------------------------------------------#
        #   所使用的YoloV7的版本。l、x
        # ---------------------------------------------------------------------#
        "phi": 'l',
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": 0.5,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": 0.3,
        # ---------------------------------------------------------------------#
        #   最大框的数量
        # ---------------------------------------------------------------------#
        "max_boxes": 100,
        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        "letterbox_image": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化yolo
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

            # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   载入模型
    # ---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.model = yolo_body([None, None, 3], self.anchors_mask, self.num_classes, self.phi)
        self.model.load_weights(self.model_path, by_name=True)

        if self.phi == "l":
            fuse_layers = [
                ["rep_conv_1", False, True],
                ["rep_conv_2", False, True],
                ["rep_conv_3", False, True],
            ]
            self.model_fuse = yolo_body([None, None, 3], self.anchors_mask, self.num_classes, self.phi, mode="predict")
            self.model_fuse.load_weights(self.model_path, by_name=True)

            fusion_rep_vgg(fuse_layers, self.model, self.model_fuse)
            del self.model
            gc.collect()
            self.model = self.model_fuse
        print('{} model, anchors, and classes loaded.'.format(model_path))
        # ---------------------------------------------------------#
        #   在DecodeBox函数中，我们会对预测结果进行后处理
        #   后处理的内容包括，解码、非极大抑制、门限筛选等
        # ---------------------------------------------------------#
        self.input_image_shape = Input([2, ], batch_size=1)
        inputs = [*self.model.output, self.input_image_shape]
        outputs = Lambda(
            DecodeBox,
            output_shape=(1,),
            name='yolo_eval',
            arguments={
                'anchors': self.anchors,
                'num_classes': self.num_classes,
                'input_shape': self.input_shape,
                'anchor_mask': self.anchors_mask,
                'confidence': self.confidence,
                'nms_iou': self.nms_iou,
                'max_boxes': self.max_boxes,
                'letterbox_image': self.letterbox_image
            }
        )(inputs)
        self.yolo_model = Model([self.model.input, self.input_image_shape], outputs)

    @tf.function
    def get_pred(self, image_data, input_image_shape):
        out_boxes, out_scores, out_classes = self.yolo_model([image_data, input_image_shape], training=False)
        return out_boxes, out_scores, out_classes

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image, crop=False, count=False):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度，并进行归一化
        # ---------------------------------------------------------#
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        # ---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        # ---------------------------------------------------------#
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape)

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        # ---------------------------------------------------------#
        #   计数
        # ---------------------------------------------------------#
        if count:
            print("top_label:", out_classes)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(out_classes == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        # ---------------------------------------------------------#
        #   是否进行目标的裁剪
        # ---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(out_boxes)):
                top, left, bottom, right = out_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        res = []
        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[int(c)]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))
            res.append([left, top, right, bottom, predicted_class, score])

        return res


yolo = YOLO()
files = os.listdir('./img/')
for file in files:
    if file.endswith('jpg') or file.endswith('png'):
        image_path = './img/' + file
        image = cv2.imread(image_path)
        items = yolo.detect_image(Image.open(image_path))

        for item in items:
            x1 = item[0]
            y1 = item[1]
            x2 = item[2]
            y2 = item[3]
            predict_class = item[4]
            score = item[5]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            label = '{} {:.2f}'.format(predict_class, score)
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)

        # cv2.putText(image, 'sum of flake: ' + str(len(items)), (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow('', image)
        cv2.waitKey(0)