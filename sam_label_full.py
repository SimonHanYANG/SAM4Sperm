'''
Author: SimonHanYANG SimonCK666@mail.163.com
Date: 2023-12-19 18:49:54
LastEditors: SimonHanYANG SimonCK666@mail.163.com
LastEditTime: 2023-12-19 21:27:11
FilePath: \Paper2-DepthEstimation_through_classification\SAM\sam_label_full.py
Description: 使用 SAM 进行 label 的完整代码
'''
import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

#============================================================================
#============================================================================
# mouse event
# 鼠标按下的事件处理
def on_mouse_press(event):
    global rect_start, drawing, current_rect
    if event.inaxes is not None and event.button == 1:  # 只在鼠标左键按下时进行处理
        rect_start = (int(event.xdata), int(event.ydata))
        drawing = True
        # 创建一个新的矩形
        current_rect = plt.Rectangle(rect_start, 0, 0, edgecolor='red', facecolor='none', lw=2)
        ax.add_patch(current_rect)
        fig.canvas.draw()

# 鼠标移动的事件处理
def on_mouse_move(event):
    if event.inaxes is not None and drawing:
        rect_end = (int(event.xdata), int(event.ydata))
        current_rect.set_width(rect_end[0] - rect_start[0])
        current_rect.set_height(rect_end[1] - rect_start[1])
        fig.canvas.draw()

# 鼠标释放的事件处理
def on_mouse_release(event):
    global rect_end, input_boxes, drawing, current_rect
    if event.inaxes is not None and event.button == 1 and drawing:
        rect_end = (int(event.xdata), int(event.ydata))
        drawing = False
        # 存储矩形框的坐标
        input_boxes.append([rect_start[0], rect_start[1], rect_end[0], rect_end[1]])
        current_rect = None  # 重置当前矩形对象
        fig.canvas.draw()

# 键盘按键的事件处理
def on_key_press(event):
    global input_boxes, current_rect
    if event.key == 'q':
        if input_boxes:
            input_boxes.pop()  # 移除最后添加的矩形框
            if current_rect:
                current_rect.remove()  # 从图像中移除矩形对象
                fig.canvas.draw()
            current_rect = None

#============================================================================
#============================================================================
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

# image: PIL Image -- seg roi ori image
# file_path: Full seg roi ori image path
# name_without_extension: seg roi ori image name --> also the mask file name

#============================================================================
#============================================================================
# CV2 read image
def cv2_read_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def show_cv2_image(image):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis('on')
    plt.show()

#============================================================================
#============================================================================
# Load SAM checkpoint
def load_sam_checkboint(sam_checkpoint, model_type, device):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print("Load SAM model DONE...")

    return sam, predictor

# sam checkpoint
sam_checkpoint = "./SAM_Checkpoints/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cpu"  # or  "cuda"
# device = "cuda"  # or  "cuda"

# Load sam checkpoint and init predictor
sam, predictor = load_sam_checkboint(sam_checkpoint, model_type, device)
predictor = SamPredictor(sam)
print("Load SAM model DONE...")

#============================================================================
#============================================================================
# 设置包含图片的文件夹路径
folder_path = "E:\\Paper2-DepthEstimation_through_classification\\seg_ori_rename\\0\\" 

# 获取文件夹中所有文件的文件名
files = os.listdir(folder_path)

# 遍历文件名，打印无后缀的图片文件名
for file_name in files:
    # 构建完整的文件路径
    file_path = os.path.join(folder_path, file_name)
    # 检查文件是否为图片
    img = Image.open(file_path)
    img.verify()  # 验证文件是否损坏
    # 分离文件名和扩展名，并打印无后缀的文件名
    name_without_extension, _ = os.path.splitext(file_name)
    print("Opened Seg ROI Image: {}.".format(file_name))
    
    # cv2 read image
    image = cv2_read_image(file_path)
    # show_cv2_image()
    
    # 通过调用SamPredictor.set_image函数，将输入的图像进行编码，
    # SamPredictor 会使用这些编码进行后续的目标分割任务。
    predictor.set_image(image)
    print("Image Embedding DONE...")
    
    # 全局变量来存储矩形框的起点和终点坐标
    rect_start = None
    rect_end = None
    input_boxes = []  # 用于存储所有矩形框的坐标
    drawing = False  # 用于标记是否正在绘制矩形
    current_rect = None  # 用于存储当前正在绘制的矩形对象
    
    # 创建图像和绘图环境
    fig, ax = plt.subplots()
    # 假设有一个名为 'image' 的图像数组已经预先加载好
    # image = cv2.imread('path_to_image.jpg') # 例子，如果您已经加载了图像
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # OpenCV 加载的图像通常是 BGR 格式，需要转换为 RGB

    # 显示图像
    ax.imshow(image)

    # 连接鼠标事件处理函数
    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)
    fig.canvas.mpl_connect('key_press_event', on_key_press)  # 连接键盘事件
    
    plt.show()

    # 当所有矩形框绘制完成后，转换为torch.tensor
    input_boxes_tensor = torch.tensor(input_boxes, device=device)  # 假设predictor.device返回的是'cuda:0'
    print(input_boxes_tensor)
    
    # 创建图像和绘图环境
    fig, ax = plt.subplots()
    # 假设有一个名为 'image' 的图像数组已经预先加载好
    # image = cv2.imread('path_to_image.jpg') # 举例，如果已经加载了图像
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # OpenCV 加载的图像需要转换为 RGB

    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes_tensor, image.shape[:2])

    import time
    start_time = time.time()  # 获取开始时间
    masks, scores, logits = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    end_time = time.time()  # 获取结束时间
    elapsed_time = end_time - start_time  # 计算经过的时间
    print(f"Segmentation spent for {elapsed_time} s.")

    print("Mask shape: {}.".format(masks.shape))  # (batch_size) x (num_predicted_masks_per_input) x H x W

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in input_boxes:
        show_box(box, plt.gca())
    plt.axis('off')
    plt.show()
    print("SAM mask DONE!")
    
    #======================================================================
    #======================================================================
    # save all masks oneto one image
    # 初始化一个全零的数组，它将保存所有的掩码
    h, w = masks[0].shape[-2:]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        # 确保掩码是正确的形状 (M, N)
        mask = mask.detach().cpu().numpy().squeeze()
        # 移除任何单一维度
        mask = mask.squeeze()
        # 确保掩码是 uint8 类型
        mask = mask.astype(np.uint8)

        # 确保掩码是二进制的 (0 或者 1)
        mask = mask > 0  # 假设mask中所有非0值表示掩码的区域
        # 累加掩码，将掩码的区域设置为255 (白色)
        combined_mask = np.where(mask, 255, combined_mask)
        
    print(combined_mask)
    
    mask_root = "..//seg_ori_mask//"
    if not os.path.exists(mask_root):
        os.makedirs(mask_root)
    
    mask_path = mask_root + file_name

    # 使用matplotlib来保存图像，不显示坐标系
    plt.axis('off')  # 不显示坐标轴
    print("Show combined mask!")
    plt.imshow(combined_mask, cmap='gray')
    plt.savefig(mask_path, bbox_inches='tight', pad_inches=0)
    print("Mask saved to {}".format(mask_path))
    plt.show()
    plt.close()  # 关闭图像，以便后续继续使用plt
    