from utils.librarys import *
from models.maskrcnn import *

class Inference_ResNet101(object):
    def __init__(self, weights="weights/resnet101_skin_segmentation_GIRL_Augment_962.pth"):
        self.weights = weights
        self.model = resnet101_maskRCNN(5)
        # self.model_detect = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device=self.device)
        self.model.load_state_dict(torch.load(self.weights, map_location=torch.device(self.device)))
        self.model.eval()

        self.label_to_num = {'skin':1}
        self.class_names = ['BG', 'skin']

    def check_type(self, img_path):
        if type(img_path) == str:
            if img_path.endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(img_path)
            else:
                raise Exception("Please input a image file")
        elif type(img_path) == np.ndarray:
            img = img_path
        return img

    def crop(self, img):
        results = self.model_detect(img)
        t = results.pandas().xyxy[0]  # or .show(), .save(), .crop(), .pandas(), etc.
        bbox = np.int32(np.array(t)[:,:4][np.where(np.array(t)[:,6] == 'person')])
        if len(bbox) > 1:
            area = (bbox[:,2] - bbox[:,0]) * (bbox[:,3] - bbox[:,1])
            bbox = bbox[np.where(max(area))]
        # Frame ma khong detect duoc thi lay ca frame luon
        elif len(bbox) < 1:
            h, w, _ = img.shape
            bbox = np.array([[0, 0, w, h]], dtype=np.uint32)

        x1, y1, x2, y2 = bbox[0]
        return x1, y1, x2, y2, img[y1:y2, x1:x2]
        
    def predict(self, img_path):
        img = self.check_type(img_path)

        # img_original = img.copy()
        # x1, y1, x2, y2, img = self.crop(img)

        img_predict = torch.as_tensor(img[:,:,::-1].transpose(2,0,1)/255.0, dtype=torch.float32).cpu()

        with torch.no_grad():
            prediction = self.model(img_predict[None, :])[0]

        labels = prediction["labels"] # tensor([1])
        scores = prediction["scores"] # tensor([0.99])
        boxes = np.int32(prediction['boxes'].cpu().numpy()) # tensor([[  x_min,   y_min,  x_max,   y_max]])
        
        masks = prediction["masks"].squeeze(1)
        masks = np.round(masks.cpu().numpy())
        final_mask = np.zeros((img.shape[0], img.shape[1]))
        for j in range(len(masks)):
            final_mask += masks[j]
        masks = np.clip(final_mask, 0, 255)

        masks = np.uint8(cv2.merge([final_mask, final_mask, final_mask]) * 255)

        # mask_original = np.zeros_like(img_original)
        # mask_original[y1:y2, x1:x2] = masks

        # color = (0, 255, 0)
        # mask_original[:,:,0][np.where(mask_original[:,:,0]==255)] = color[0]
        # mask_original[:,:,1][np.where(mask_original[:,:,1]==255)] = color[1]
        # mask_original[:,:,2][np.where(mask_original[:,:,2]==255)] = color[2]

        # img = cv2.addWeighted(img_original, 0.85, mask_original, 0.4, 0)
        img = cv2.addWeighted(img, 0.85, masks, 0.4, 0)

        return img

#---------------------------------------------------------------------------------------------------------------------
ResNet101 = Inference_ResNet101(weights='weights/resnet101_skin_segmentation_GIRL_Augment_962.pth')

def image(path_img):
    print('Processing image... \nSaved_folder in datasets/results/...')
    img = ResNet101.predict(path_img)
    cv2.imwrite('datasets/results/' + path_img.split('/')[-1], img)

def video(path_video='datasets/dathao1.mp4'):
    print('Processing video... \n Please wait...')
    cap = cv2.VideoCapture(path_video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    fps = 30
    out = cv2.VideoWriter('results_' + path_video.split('/')[-1], cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)

    while True:
        _, frame = cap.read()
        try:
            frame = ResNet101.predict(frame)
            out.write(frame)
        except:
            out.release()
            break
    out.release()
    print('Done!')

def webcam():
    print("Using webcam, press q to exit, press s to save")
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        start = time.time()
        frame = ResNet101.predict(frame)
        # FPS
        fps = round(1 / (time.time() - start), 2)
        cv2.putText(frame, "FPS : " + str(fps), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.imshow('Prediction', frame)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('s'):
            cv2.imwrite('datasets/results/' + str(time.time()) + '.jpg', frame)
        if k == ord('q'):
            break

if __name__ == '__main__':
    image('5body.jpg')
    # video('datasets/dathao1.mp4')
    # webcam()

