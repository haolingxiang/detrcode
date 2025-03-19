import cv2
import numpy as np


def draw_rotated_bbox(image_path, txt_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("Error: 无法读取图像文件")
        return

    # 读取目标框信息
    with open(txt_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            if len(data) < 9:
                continue

            # 解析四个顶点坐标
            points = [
                (int(data[0]), int(data[1])),
                (int(data[2]), int(data[3])),
                (int(data[4]), int(data[5])),
                (int(data[6]), int(data[7]))
            ]
            label = data[8]

            if label == "apron":
                # 绘制斜框
                pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0),
                              thickness=2, lineType=cv2.LINE_AA)

                # 添加标签（带背景框）
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(image,
                              (points[0][0], points[0][1] - text_h - 10),
                              (points[0][0] + text_w, points[0][1]),
                              (0, 255, 0), -1)
                cv2.putText(image, label,
                            (points[0][0], points[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 保存结果
    cv2.imwrite(output_path, image)
    print(f"结果已保存至: {output_path}")


# 使用示例
if __name__ == "__main__":
    draw_rotated_bbox('.././my_detr/star/images/0006.png',
                      '.././my_detr/star/obj_txt/0006.txt',
                      'output_rotated_1.jpg')