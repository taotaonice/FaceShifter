import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from mtcnn_pytorch.src.get_nets import PNet, RNet, ONet
from mtcnn_pytorch.src.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from mtcnn_pytorch.src.first_stage import run_first_stage
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class MTCNN():
    def __init__(self):
        self.pnet = PNet().to(device)
        self.rnet = RNet().to(device)
        self.onet = ONet().to(device)
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
        self.refrence = get_reference_facial_points(default_square= True)
        
    def align(self, img, crop_size=(112, 112), return_trans_inv=False):
        _, landmarks = self.detect_faces(img)
        if len(landmarks) == 0:
            return None if not return_trans_inv else (None, None)
        facial5points = [[landmarks[0][j],landmarks[0][j+5]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=crop_size,
                                         return_trans_inv=return_trans_inv)
        if return_trans_inv:
            return Image.fromarray(warped_face[0]), warped_face[1]
        else:
            return Image.fromarray(warped_face)

    def align_fully(self, img, crop_size=(112, 112), return_trans_inv=False, ori=[0, 1, 3], fast_mode=True):
        ori_size = img.copy()
        h = img.size[1]
        w = img.size[0]
        sw = 320. if fast_mode else w
        scale = sw / w
        img = img.resize((int(w*scale), int(h*scale)))
        candi = []
        for i in ori:
            if len(candi) > 0:
                break
            if i > 0:
                rimg = img.transpose(i+1)
            else:
                rimg = img
            box, landmarks = self.detect_faces(rimg, min_face_size=sw/10, thresholds=[0.6, 0.7, 0.7])
            landmarks /= scale
            if len(landmarks) == 0:
                continue
            if i == 0:
                f5p = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
            elif i == 1:
                f5p = [[w-1-landmarks[0][j+5], landmarks[0][j]] for j in range(5)]
            elif i == 2:
                f5p = [[w-1-landmarks[0][j], h-1-landmarks[0][j+5]] for j in range(5)]
            elif i == 3:
                f5p = [[landmarks[0][j + 5], h-1-landmarks[0][j]] for j in range(5)]
            candi.append((box[0][4], f5p))
        if len(candi) == 0:
            return None if not return_trans_inv else (None, None)
        while len(candi) > 1:
            if candi[0][0] > candi[1][0]:
                del candi[1]
            else:
                del candi[0]
        facial5points = candi[0][1]
        warped_face = warp_and_crop_face(np.array(ori_size), facial5points, self.refrence, crop_size=crop_size,
                                         return_trans_inv=return_trans_inv)
        if return_trans_inv:
            return Image.fromarray(warped_face[0]), warped_face[1]
        else:
            return Image.fromarray(warped_face)

    def align_multi(self, img, limit=None, min_face_size=64.0, crop_size=(112, 112)):
        boxes, landmarks = self.detect_faces(img, min_face_size)
        if len(landmarks) == 0:
            return None
        if limit:
            boxes = boxes[:limit]
            landmarks = landmarks[:limit]
        faces = []
        for landmark in landmarks:
            facial5points = [[landmark[j],landmark[j+5]] for j in range(5)]
            warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=crop_size)
            faces.append(Image.fromarray(warped_face))
        # return boxes, faces
        return faces

    def get_landmarks(self, img, min_face_size=32, crop_size=(256, 256), fast_mode=False, ori=[0,1,3]):
        ori_size = img.copy()
        h = img.size[1]
        w = img.size[0]
        sw = 640. if fast_mode else w
        scale = sw / w
        img = img.resize((int(w*scale), int(h*scale)))
        min_face_size = min_face_size if not fast_mode else sw/20
        candi = []
        boxes = np.zeros([0, 5])
        for i in ori:
            if i > 0:
                rimg = img.transpose(i+1)
            else:
                rimg = img
            box, landmarks = self.detect_faces(rimg, min_face_size=min_face_size, thresholds=[0.6, 0.7, 0.7])
            landmarks /= scale
            if len(landmarks) == 0:
                continue
            if i == 0:
                f5p = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
            elif i == 1:
                f5p = [[w-1-landmarks[0][j+5], landmarks[0][j]] for j in range(5)]
                x1 = w-1-box[:, 1]
                y1 = box[:, 0]
                x2 = w-1-box[:, 3]
                y2 = box[:, 2]
                box[:, :4] = np.stack((x2, y1,  x1, y2), axis=1)
            elif i == 2:
                f5p = [[w-1-landmarks[0][j], h-1-landmarks[0][j+5]] for j in range(5)]
                x1 = w-1-box[:, 0]
                y1 = h-1-box[:, 1]
                x2 = w-1-box[:, 2]
                y2 = h-1-box[:, 3]
                box[:, :4] = np.stack((x2, y2,  x1, y1), axis=1)
            elif i == 3:
                f5p = [[landmarks[0][j + 5], h-1-landmarks[0][j]] for j in range(5)]
                x1 = box[:, 1]
                y1 = h-1-box[:, 0]
                x2 = box[:, 3]
                y2 = h-1-box[:, 2]
                box[:, :4] = np.stack((x1, y2, x2, y1), axis=1)
            candi.append(f5p)
            boxes = np.concatenate((boxes, box), axis=0)
        # pick = nms(boxes)
        faces = []
        for idx, facial5points in enumerate(candi):
            # if idx not in pick:
            #     continue
            warped_face = warp_and_crop_face(np.array(ori_size), facial5points, self.refrence, crop_size=crop_size,
                                             return_trans_inv=False)
            faces.append((warped_face, facial5points))
        return faces

    def detect_faces(self, image, min_face_size=64.0,
                     thresholds=[0.6, 0.7, 0.8],
                     nms_thresholds=[0.7, 0.7, 0.7]):
        """
        Arguments:
            image: an instance of PIL.Image.
            min_face_size: a float number.
            thresholds: a list of length 3.
            nms_thresholds: a list of length 3.

        Returns:
            two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
            bounding boxes and facial landmarks.
        """

        # BUILD AN IMAGE PYRAMID
        width, height = image.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size/min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m*factor**factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1

        # it will be returned
        bounding_boxes = []

        with torch.no_grad():
            # run P-Net on different scales
            for s in scales:
                boxes = run_first_stage(image, self.pnet, scale=s, threshold=thresholds[0])
                bounding_boxes.append(boxes)

            # collect boxes (and offsets, and scores) from different scales
            bounding_boxes = [i for i in bounding_boxes if i is not None]
            if len(bounding_boxes) == 0:
                return np.zeros([0]), np.zeros([0])
            bounding_boxes = np.vstack(bounding_boxes)

            keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
            bounding_boxes = bounding_boxes[keep]

            # use offsets predicted by pnet to transform bounding boxes
            bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
            # shape [n_boxes, 5]

            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            # STAGE 2

            img_boxes = get_image_boxes(bounding_boxes, image, size=24)
            img_boxes = torch.FloatTensor(img_boxes).to(device)

            output = self.rnet(img_boxes)
            offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > thresholds[1])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]

            keep = nms(bounding_boxes, nms_thresholds[1])
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            # STAGE 3

            img_boxes = get_image_boxes(bounding_boxes, image, size=48)
            if len(img_boxes) == 0:
                return np.zeros([0]), np.zeros([0])
            img_boxes = torch.FloatTensor(img_boxes).to(device)
            output = self.onet(img_boxes)
            landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
            offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > thresholds[2])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]
            landmarks = landmarks[keep]

            # compute landmark points
            width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
            height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
            xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
            landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
            landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

            bounding_boxes = calibrate_box(bounding_boxes, offsets)
            keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
            bounding_boxes = bounding_boxes[keep]
            landmarks = landmarks[keep]

        return bounding_boxes, landmarks
