import os
import imagesize


def find_large_images(folder_path, min_pixels):
    large_images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):
                file_path = os.path.join(root, file)
                try:
                    width, height = imagesize.get(file_path)
                    if width * height > min_pixels:
                        large_images.append((file_path, width, height))
                except Exception as e:
                    print(f"解析失败：{file_path}，错误：{str(e)}")
    return large_images


if __name__ == "__main__":
    # 配置参数
    folder_path = r"F:\dota\train\images"  # 替换为目标文件夹路径
    min_pixels = 178956970

    # 执行筛选
    result = find_large_images(folder_path, min_pixels)

    # 输出结果
    print(f"找到 {len(result)} 张符合条件的图片：")
    for img in result:
        print(f"路径：{img[0]}，分辨率：{img[1]}x{img[2]}，总像素：{img[1] * img[2]:,}")
