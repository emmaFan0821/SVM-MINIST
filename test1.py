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
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
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
pic_num = 1
img,_ = dataset_test[pic_num-1]
model = torch.load('\model.pkl')
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])
print('pic', pic_num)
print('prediction\n', prediction)
print(type(prediction))

score_thresh = 0.9
img1 = Image.open("/Data_HDD/INT408_20/INT408_1/Siyuan_Fan/INT408_a2/PennFudanPed/PNGImages/FudanPed00001.png")
for i in range(prediction[0]['boxes'].shape[0]):
    if prediction[0]['scores'][i] > score_thresh:
        a = ImageDraw.ImageDraw(img1)
        x_draw_min = round(prediction[0]['boxes'][i][0].item())
        y_draw_min = round(prediction[0]['boxes'][i][1].item())
        x_draw_max = round(prediction[0]['boxes'][i][2].item())
        y_draw_max = round(prediction[0]['boxes'][i][3].item())
        a.rectangle((x_draw_max, y_draw_max, x_draw_min, y_draw_min), fill=None, outline='red', width=7)
        score = str(prediction[0]['scores'][i].item())
        a.text((x_draw_min, y_draw_min), score)

img1.save("/Data_HDD/INT408_20/INT408_1/Siyuan_Fan/INT408_a2/img01.jpg")
