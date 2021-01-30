from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from engine import train_one_epoch, evaluate
import utils
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import os
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import transforms as T
from PIL import Image
from PIL import ImageDraw

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())  # 将dataset类中__getitem__()方法内读入的PIL或CV的图像数据转换为torch.FloatTensor
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))  # 依据概率p对PIL图片进行水平翻转
    return T.Compose(transforms)


def get_instance_segmentation_model(num_classes):

    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 256
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    backbone.add_module('conv',torch.nn.Conv2d(1280,256,kernel_size=(1,1),stride=(1,1),bias=False))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))
pic_num =13
img,_ = dataset_test[pic_num-1]  # model processes pic
model = torch.load('\model.pkl')
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])  # get prediction

print('prediction\n', prediction)

img1 = Image.open("/Data_HDD/INT408_20/INT408_1/Siyuan_Fan/INT408_a2/PennFudanPed/PNGImages/FudanPed00013.png")
score_thresh = 0.9
IoU_thresh = 0.94
X_true = [[389, 554]]  # fig13
Y_true = [[193, 476]]
# X_true = [[160, 302], [420, 535]]  # fig1
# Y_true = [[182, 431], [171, 486]]
# X_true = [[270, 380], [35, 135], [159, 252]]  # fig59
# Y_true = [[45, 328], [54, 330], [36, 329]]
print('pic', pic_num)
for i in range(prediction[0]['boxes'].shape[0]):
    x_draw_min = round(prediction[0]['boxes'][i][0].item())  # get coordinates
    y_draw_min = round(prediction[0]['boxes'][i][1].item())
    x_draw_max = round(prediction[0]['boxes'][i][2].item())
    y_draw_max = round(prediction[0]['boxes'][i][3].item())
    if prediction[0]['scores'][i] > score_thresh:  # whether output the bbox
        a = ImageDraw.ImageDraw(img1)
        x_true_min = X_true[i][0]
        y_true_min = Y_true[i][0]
        x_true_max = X_true[i][1]
        y_true_max = Y_true[i][1]
        # calculate overlap area
        width_max = max(x_true_max, x_draw_max) - min(x_true_min, x_draw_min)
        width_true = x_true_max - x_true_min
        width_draw = x_draw_max - x_draw_min
        width = width_true + width_draw - width_max

        height_max = max(y_true_max, y_draw_max) - min(y_true_min, y_draw_min)
        height_true = y_true_max - y_true_min
        height_draw = y_draw_max - y_draw_min
        height = height_true + height_draw - height_max

        s_true = width_true * height_true
        s_draw = width_draw * height_draw
        s_overlay = height * width
        IoU = s_overlay / (s_true + s_draw - s_overlay)
        print('IoU:', IoU)
        score = str(prediction[0]['scores'][i].item())

        if IoU < IoU_thresh:
            print('Inaccuracy of results')
            a.rectangle((x_true_max, y_true_max, x_true_min, y_true_min), fill=None, outline='black', width=5)
            a.rectangle((x_draw_max, y_draw_max, x_draw_min, y_draw_min), fill=None, outline='red', width=7)
            a.text((x_draw_min, y_draw_min), score)
        else:
            a.rectangle((x_draw_max, y_draw_max, x_draw_min, y_draw_min), fill=None, outline='green', width=5)
            a.text((x_draw_min, y_draw_min), score)

img1.save("/Data_HDD/INT408_20/INT408_1/Siyuan_Fan/INT408_a2/img13-mobilenet.jpg")
