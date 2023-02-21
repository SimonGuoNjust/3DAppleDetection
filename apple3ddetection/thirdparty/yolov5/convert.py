import os
import numpy as np
import torch
from PIL import Image

#####################################
# Class that takes the input instance masks
# and extracts bounding boxes on the fly
#####################################

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

class AppleDataset(object):
    def __init__(self, root_dir, save_path, transforms):
        self.root_dir = root_dir
        self.transforms = transforms
        self.save_path = save_path

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        # Load all image and mask files, sorting them to ensure they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root_dir, "masks"))))

    def __getitem__(self, idx):
        # Load images and masks
        img_path = os.path.join(self.root_dir, "images", self.imgs[idx])
        mask_path = os.path.join(self.root_dir, "masks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)     # Each color of mask corresponds to a different instance with 0 being the background

        # Convert the PIL image to np array
        mask = np.array(mask)
        obj_ids = np.unique(mask)

        # Remove background id
        obj_ids = obj_ids[1:]

        # Split the color-encoded masks into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # Get bbox coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        h, w = mask.shape
        for ii in range(num_objs):
            pos = np.where(masks[ii])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            if xmin == xmax or ymin == ymax:
                continue

            xmin = np.clip(xmin, a_min=0, a_max=w)
            xmax = np.clip(xmax, a_min=0, a_max=w)
            ymin = np.clip(ymin, a_min=0, a_max=h)
            ymax = np.clip(ymax, a_min=0, a_max=h)

            width = (xmax-xmin) / w
            height = (ymax-ymin) / h
            cx = (xmin + xmax) / 2.0 / w
            cy = (ymin + ymax) / 2.0 / h

            boxes.append([cx,cy,width,height])

        with open(os.path.join(self.save_path,self.imgs[idx].split('.')[0]+'.txt'),'w') as f:
            for box in boxes:
                line = f"0 {box[0]} {box[1]} {box[2]} {box[3]}\n"
                f.write(line)
        # print(self.imgs[idx],boxes)

        img = torchvision.transforms.ToTensor()(img)

        # # Convert everything into a torch.Tensor
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        #
        # # There is only one class (apples)
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        # masks = torch.as_tensor(masks, dtype=torch.uint8)
        #
        # image_id = torch.tensor([idx])
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        #
        # # All instances are not crowd
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        #
        # target = {}
        # target["boxes"] = boxes
        # target["labels"] = labels
        # target["masks"] = masks
        # target["image_id"] = image_id
        # target["area"] = area
        # target["iscrowd"] = iscrowd
        #
        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

        return img

    def __len__(self):
        return len(self.imgs)

    def get_img_name(self, idx):
        return self.imgs[idx]

if __name__ == "__main__":
    import torchvision,tqdm
    dataset = AppleDataset(os.path.join('/home/psdz/dataset/apple_dataset/v1.0/', 'train'),'/home/psdz/dataset/apple_dataset/v1.0/train/labels',torchvision.transforms.ToTensor())

    print("Creating data loaders")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=True,
                                              num_workers=12,)

    for _ in tqdm.tqdm(data_loader):
        pass
        # print("Epoch - 1")