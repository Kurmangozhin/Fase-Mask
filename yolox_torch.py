import cv2, os, random, colorsys, onnxruntime, time, functools
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import time, argparse, uuid, logging




parser = argparse.ArgumentParser("YOLOX Inference")
parser.add_argument('--i', type = str, required = True, default = False)




def display_process_time(func):
    @functools.wraps(func)
    def decorated(*args, **kwargs):
        s1 = time.time()
        res = func(*args, **kwargs)
        s2 = time.time()
        print('%s process time %f s' % (func.__name__, (s2-s1)/60))
        return res
    return decorated


providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]




class Processing(object):
	def __init__(self):
		pass

	def cvtColor(self, image):
	    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
	        return image 
	    else:
	        image = image.convert('RGB')
	        return image 

	def get_classes(self, classes_path):
	    with open(classes_path, encoding='utf-8') as f:
	        class_names = f.readlines()
	    class_names = [c.strip() for c in class_names]
	    return class_names, len(class_names)
	

	def preprocess_input(self, image):
	    image_data = image / 255.0
	    image_data -= np.array([0.485, 0.456, 0.406])
	    image_data /= np.array([0.229, 0.224, 0.225])
	    return image_data


	def resize_image(self, image, size):
	    iw, ih  = image.size
	    w, h    = size
	    scale   = min(w/iw, h/ih)
	    nw      = int(iw*scale)
	    nh      = int(ih*scale)
	    image   = image.resize((nw,nh), Image.BICUBIC)
	    new_image = Image.new('RGB', size, (128,128,128))
	    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
	    return new_image


	def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
	    box_yx = box_xy[..., ::-1]
	    box_hw = box_wh[..., ::-1]
	    input_shape = np.array(input_shape)
	    image_shape = np.array(image_shape)
	    new_shape = np.round(image_shape * np.min(input_shape/image_shape))
	    offset  = (input_shape - new_shape)/2./input_shape
	    scale   = input_shape/new_shape
	    box_yx  = (box_yx - offset) * scale
	    box_hw *= scale
	    box_mins    = box_yx - (box_hw / 2.)
	    box_maxes   = box_yx + (box_hw / 2.)
	    boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
	    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
	    return boxes 


	def colors(self, class_labels):
	    hsv_tuples = [(x / len(class_labels), 1., 1.) for x in range(len(class_labels))]
	    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	    class_colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
	    np.random.seed(43)
	    np.random.shuffle(colors)
	    np.random.seed(None)
	    class_colors = np.tile(class_colors, (16, 1))
	    return class_colors




class Detection(Processing):
    def __init__(self, path_model, path_classes, image_shape):
        self.session = onnxruntime.InferenceSession(path_model, providers = providers)
        self.class_labels, self.num_names = self.get_classes(path_classes)
        self.image_shape = image_shape
        self.font = ImageFont.truetype('font.otf', 10)
        self.class_colors = self.colors(self.class_labels)


    def postprocess(self, outputs, image_shape):
        box_xy, box_wh      = (outputs[:, 0:2] + outputs[:, 2:4])/2, outputs[:, 2:4] - outputs[:, 0:2]
        outputs[:, :4]    = self.yolo_correct_boxes(box_xy, box_wh, self.image_shape, image_shape)
        return outputs
    
        
    def decode_boxes(self, outputs, image_shape):
        outputs = self.postprocess(outputs, image_shape)
        box_out = outputs[:, :4]
        scores_out = outputs[:, 4] * outputs[:, 5]
        classes_out = np.array(outputs[:, 6], dtype = 'int32')
        return box_out, scores_out, classes_out


    def inference(self, input, image_shape):
        ort_inputs = {self.session.get_inputs()[0].name:input[None,:,:,:]}
        outputs = self.session.run(None, ort_inputs)[0]
        box_out, scores_out, classes_out = self.decode_boxes(outputs, image_shape)
        return box_out, scores_out, classes_out


    def draw_detection(self, image, boxes_out, scores_out, classes_out):
        image_pred = image.copy()
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.image_shape), 1))
        for i, c in reversed(list(enumerate(classes_out))):
            draw = ImageDraw.Draw(image_pred)
            predicted_class = self.class_labels[c]
            box = boxes_out[i]
            score = scores_out[i]
            label = '{}:{:.2f}%'.format(predicted_class, score*100)
            label_size = draw.textsize(label, self.font)
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            if top - label_size[1] >= 0:
                  text_origin = np.array([left, top - label_size[1]])
            else:
                  text_origin = np.array([left, top + 1])
            draw.rectangle([left, top, right, bottom], outline= tuple(self.class_colors[c]), width=2)
            draw.text(text_origin, label, fill = (255,255,0), font = self.font)      
            del draw
        return np.array(image_pred)


    def __call__(self, image_path:str):
        image = Image.open(image_path).convert("RGB")
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)       
        image = self.cvtColor(image)
        image_data  = self.resize_image(image, (self.image_shape[1], self.image_shape[0]))
        image_data  = np.transpose(self.preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1))    
        box_out, scores_out, classes_out = self.inference(image_data, input_image_shape)
        image_pred = self.draw_detection(image, box_out, scores_out, classes_out)
        return np.array(image_pred)



if __name__ == '__main__':
    args_parser = parser.parse_args()
    args = {"path_model":"mask.onnx", "path_classes":'classes.txt', "image_shape":(640, 640)}
    detector = Detection(**args)
    image_pred = detector(args_parser.i)
    image = cv2.cvtColor(image_pred, cv2.COLOR_BGR2RGB)
    cv2.imwrite("out.jpg", image)

