import os
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from tqdm import tqdm
device = torch.device('cuda')

mtcnn = MTCNN(image_size=(1280, 720), device=device)
root_dir = '/scratch/chumache/dfer_datasets/mfaw/clips/'
write_dir = '/scratch/chumache/dfer_datasets/mfaw/clips_faces/'

if True:
    for filename in os.listdir(root_dir):
        os.mkdir(os.path.join(write_dir, filename.split('.')[0]))
        cap = cv2.VideoCapture(os.path.join(root_dir, filename))
        if not cap.isOpened():
            print(f"Error: Unable to open video file '{video_path}'")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Loop through each frame
        frame_count = 0
        while True:
            ret, im = cap.read()
            if not ret:
                break

            # Save frame
            temp = im[:,:,-1]
            im_rgb = im.copy()
            im_rgb[:,:,-1] = im_rgb[:,:,0]
            im_rgb[:,:,0] = temp
            im_rgb = torch.tensor(im_rgb)
            im_rgb = im_rgb.to(device)
            
            bbox = mtcnn.detect(im_rgb)
            if bbox[0] is not None:

                xs = [x[0] for x in bbox[0]]
                ys = [x[1] for x in bbox[0]]
                x2s = [x[2] for x in bbox[0]]
                y2s = [x[3] for x in bbox[0]]
                bbox = [min(xs), min(ys), max(x2s), max(y2s)]

                bbox = [round(x) for x in bbox]
                x1, y1, x2, y2 = bbox
                y1 = max(0, y1)
                x1 = max(0,x1)
                w = y2 - y1
                h = x2 - x1
                if w > h:
                    diff = w - h
                    x2 += diff // 2 
                    x1 -= diff // 2
                elif h > w:
                    diff = h - w
                    y2 += diff // 2
                    y1 -= diff // 2
                im = im[max(0,y1):y2, max(0,x1):x2, :]
                cv2.imwrite(os.path.join(write_dir, filename.split('.')[0], str(frame_count) + '.jpg'), im)
                frame_count += 1
            else:
                ss = min(im.shape[0]//2, im.shape[1]//2)
                im = im[im.shape[0]//2-ss//2:im.shape[0]//2+ss//2, im.shape[1]//2-ss//2:im.shape[1]//2+ss//2,:]
                cv2.imwrite(os.path.join(write_dir, filename.split('.')[0], str(frame_count) + '.jpg'), im)
                frame_count += 1
	 
        print(filename, frame_count)
        # Release the capture
        cap.release()

