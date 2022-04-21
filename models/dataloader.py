from utils.librarys import *

class Skin_dataset(torch.utils.data.Dataset):
    def __init__(self, list_data, transforms=None):

        self.list_data = list_data
        self.transforms = transforms

    def __getitem__(self, idx):

        # load images ad labels
        img_path = self.list_data[idx][0]
        points = self.list_data[idx][1]
        img = Image.open(img_path).convert("RGB")
        
        masks = np.zeros((1, img.size[1], img.size[0]), dtype=np.uint8)

        for contour in points:
            c = np.array(contour)
            cv2.fillPoly(masks[0], pts = [c], color = 1)
            
        labels = [1]

        # get bounding box coordinates for each mask
        num_objs = len(masks)
        boxes = []
        for i in range(num_objs):

            if np.all(masks[i]==0):
                boxes.append([0, 0, 0, 0])
                continue
                
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.list_data)