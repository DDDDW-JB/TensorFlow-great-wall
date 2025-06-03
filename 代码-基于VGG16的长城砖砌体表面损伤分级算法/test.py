import tensorflow as tf
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# 创建输入数据生成器
def get_data_generators():
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # 加载测试数据
    test_generator = test_datagen.flow_from_directory(
        './test',  # 数据集路径
        target_size=(128,320),
        batch_size=1,
        class_mode='categorical',
        shuffle=True)
    return test_generator

def add_label_banner(image, label, banner_height=40):
    h, w = image.shape[:2]

    # 新图像：比原图高出 banner_height
    bannered = np.zeros((h + banner_height, w, 3), dtype=np.uint8)
    bannered[banner_height:, :, :] = image  # 原图贴到底部

    # 绘制上方线框
    margin = 5
    box_x1 = margin
    box_y1 = (banner_height - 30) // 2
    box_x2 = w - margin
    box_y2 = box_y1 + 30

    # 黑边白底线框
    cv2.rectangle(bannered, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)

    # 计算文字位置（居中）
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_x = (w - text_width) // 2
    text_y = box_y1 + text_height + 2

    # 绘制文字（红色）
    cv2.putText(bannered, label, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

    return bannered

# 推理与保存结果
def infer_and_save(model, test_generator):
    for i in range(10):  # 修改数字来设置推理图片数量
        X_batch, _ = test_generator.next()
        out = model.predict(X_batch)
        pred = np.argmax(out, axis=1)

        # 根据推理结果进行分类
        label = 'good' if pred[0] == 1 else 'damaged'
        src = np.array(X_batch[0] * 255, dtype=np.uint8)
        src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)

        result = add_label_banner(src, label)
        cv2.imwrite('test_{}.jpg'.format(i), result)
        print(f"Image {i} saved with label: {label}")

def train():
    model = tf.keras.models.load_model('best_model.h5')
    print("Model loaded from best_model.h5")

    # 加载测试数据
    test_generator = get_data_generators()
    # 进行推理并保存结果
    infer_and_save(model, test_generator)

if __name__ == '__main__':
    train()