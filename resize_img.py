from PIL import Image
import os

# 设置文件夹路径和目标宽度
folder_path = 'images/'  # 替换为你文件夹的路径
target_width = 1000  # 你希望的统一宽度

# 获取文件夹中的所有图片文件
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # 根据你的图片格式添加相应的后缀
        img_path = os.path.join(folder_path, filename)
        
        # 打开图片
        with Image.open(img_path) as img:
            # 获取图片的原始宽度和高度
            original_width, original_height = img.size
            
            # 计算按目标宽度缩放后的高度
            aspect_ratio = original_height / original_width
            new_height = int(target_width * aspect_ratio)
            
            # 调整图片大小
            img_resized = img.resize((target_width, new_height))
            
            # 保存修改后的图片
            img_resized.save(os.path.join(folder_path, f"resized_{filename}"))

print("所有图片已处理完成！")