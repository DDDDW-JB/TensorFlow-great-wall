import os
from PIL import Image

# 支持的图片扩展名
SUPPORTED_FORMATS = ('.jpg')

# 获取当前文件夹路径
current_dir = os.getcwd()

# 遍历当前文件夹中的所有文件
for filename in os.listdir(current_dir):
    if filename.lower().endswith(SUPPORTED_FORMATS):
        # 打开图像
        image_path = os.path.join(current_dir, filename)
        with Image.open(image_path) as img:
            # 旋转180度
            rotated_img = img.rotate(180)

            # 构造新文件名
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_rotated{ext}"
            new_image_path = os.path.join(current_dir, new_filename)

            # 保存翻转后的图像
            rotated_img.save(new_image_path)

print("图像翻转完成，数据增强已完成。")