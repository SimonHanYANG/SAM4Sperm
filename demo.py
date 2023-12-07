import numpy as np
import matplotlib.pyplot as plt
import cv2


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


image = cv2.imread('E:\\Paper2-DepthEstimation_through_classification\\SAM\\hs73_data\eval\\0\\roi_3.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


plt.figure(figsize=(10,10))
plt.imshow(image)
plt.axis('on')
plt.show()


import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "./SAM_Checkpoints/sam_vit_b_01ec64.pth"
# sam_checkpoint = "./SAM_Checkpoints/sam_vit_h_4b8939.pth"

model_type = "vit_b"

device = "cpu"  # or  "cuda"
# device = "cuda"  # or  "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
print("Load SAM model DONE...")

# 通过调用SamPredictor.set_image函数，将输入的图像进行编码，
# SamPredictor 会使用这些编码进行后续的目标分割任务。

predictor.set_image(image)
print("Image Embedding DONE...")

# 全局变量来存储矩形框的起点和终点坐标
rect_start = None
rect_end = None
input_box = None
drawing = False  # 用于标记是否正在绘制矩形
current_rect = None  # 用于存储当前正在绘制的矩形对象

# 鼠标按下的事件处理
def on_mouse_press(event):
    global rect_start, drawing, current_rect
    if event.inaxes is not None and event.button == 1:  # 只在鼠标左键按下时进行处理
        rect_start = (int(event.xdata), int(event.ydata))
        drawing = True
        # 如果已经有一个矩形在画，则移除它
        if current_rect:
            current_rect.remove()
        # 创建一个新的矩形
        current_rect = plt.Rectangle(rect_start, 0, 0, edgecolor='red', facecolor='none', lw=2)
        ax.add_patch(current_rect)
        fig.canvas.draw()

# 鼠标移动的事件处理
def on_mouse_move(event):
    global rect_start, rect_end, drawing, current_rect
    if event.inaxes is not None and drawing:
        rect_end = (int(event.xdata), int(event.ydata))
        current_rect.set_width(rect_end[0] - rect_start[0])
        current_rect.set_height(rect_end[1] - rect_start[1])
        fig.canvas.draw()

# 鼠标释放的事件处理
def on_mouse_release(event):
    global rect_start, rect_end, input_box, drawing, current_rect
    if event.inaxes is not None and event.button == 1:
        rect_end = (int(event.xdata), int(event.ydata))
        drawing = False
        # 更新最终的矩形框
        current_rect.set_width(rect_end[0] - rect_start[0])
        current_rect.set_height(rect_end[1] - rect_start[1])
        fig.canvas.draw()
        # 存储矩形框的坐标
        input_box = np.array([rect_start[0], rect_start[1], rect_end[0], rect_end[1]])

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

plt.show()

print(input_box)

# 创建图像和绘图环境
fig, ax = plt.subplots()
# 假设有一个名为 'image' 的图像数组已经预先加载好
# image = cv2.imread('path_to_image.jpg') # 举例，如果已经加载了图像
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # OpenCV 加载的图像需要转换为 RGB

masks, scores, logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
)

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks[0], plt.gca())
show_box(input_box, plt.gca())
plt.axis('off')
plt.show()


# save masks
for i, (mask, score) in enumerate(zip(masks, scores)):
    mask = mask + 255
    plt.imshow(mask, cmap='gray')
    plt.savefig('pic-{}.png'.format(i + 1))
    plt.show()

# save masked image
for i, (mask, score) in enumerate(zip(masks, scores)):
    mask = ~mask
    mask = mask + 255
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    mask = mask.astype(np.uint8)
    res = cv2.bitwise_and(image, mask)
    res[res == 0] = 255
    plt.imshow(res)
    plt.savefig('res-{}.png'.format(i + 1))
    plt.show()
