import colorsys
import os
from re import X
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2

import time
import shutil

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


from yolo3.model import box_iou, yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import image_preporcess

XML_FORMAT="""<annotation>
	<folder>{resim_dosyasi}</folder>
	<filename>{i}</filename>
	<path>{resim_tam_uzanti}/{i}</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>854</width>
		<height>480</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>uav</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{x_min}</xmin>
			<ymin>{y_min}</ymin>
			<xmax>{x_max}</xmax>
			<ymax>{y_max}</ymax>
		</bndbox>
	</object>
</annotation>
"""

class YOLO(object):
    _defaults = {
        #"model_path": 'logs/trained_weights_final.h5',
        "model_path": 'model_data/ep021-loss14.058-val_loss14.087.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/predefined_classes.txt',
        "score" : 0.6,
        "iou" : 0.4,
        "model_image_size" : (320,320),
        "text_size" : 1,
    }
    list=[]
    resim_dosyasi="images"
    resim_tam_uzanti=f'{os.getcwd()}/images' # "/home/predator/Desktop/otmatik_işaretleme/images"
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs) 
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

       #Modeli oluşturan ağırlıkları yükleme 
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = image_preporcess(np.copy(image), tuple(reversed(self.model_image_size)))
            image_data = boxed_image

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],#[image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        thickness = (image.shape[0] + image.shape[1]) // 600
        fontScale=1
        ObjectsList = []
        
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            #label = '{}'.format(predicted_class)
            scores = '{:.2f}'.format(score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

            mid_h = (bottom-top)/2+top
            mid_v = (right-left)/2+left

            # put object rectangle
            cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], thickness)

            # get text size
            (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, 1)

            # put text rectangle
            cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), self.colors[c], thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, (0, 0, 0), 1)

            # add everything to list
            ObjectsList.append([top, left, bottom, right, mid_v, mid_h, label, scores])

        return image, ObjectsList

    def close_session(self):
        self.sess.close()

    def detect_img(self, image):
        image = cv2.imread("images/{}".format(image), cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image_color = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        r_image, ObjectsList = self.detect_image(original_image_color)
        return r_image, ObjectsList
    def xml(self,x_min,y_min,x_max,y_max,i):
        with open(f'{image.split(".")[0]}.xml','w') as f:
            f.write(XML_FORMAT.format(
                resim_dosyasi=self.resim_dosyasi,
                resim_tam_uzanti=self.resim_tam_uzanti,
                i=image,
                x_min=str(x_min),
                x_max=str(x_max),
                y_min=str(y_min),
                y_max=str(y_max),
            ))
        print('XML olusturuldu...')

    def dosya_okuma(self):
        dosyalar = os.listdir("images")
        
        for dosya in dosyalar:
            if dosya.endswith(".jpg"):
                self.list.append(f"{dosya}")
                
        return self.list

    
if __name__=="__main__":
    yolo = YOLO()
    a=yolo.dosya_okuma()
    a.sort()
 
    for image in a:
        print(image)
        r_image, ObjectsList = yolo.detect_img(image)
        
        if not ObjectsList:
            # shutil.move("images/{}".format(i),"failDetectIMG")
            # cv2.destroyAllWindows()
            cv2.imshow(image, r_image)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                cv2.destroyAllWindows()

            elif cv2.waitKey(0) & 0xFF == ord("a"): #sonradan tespit edilcek resimler
                shutil.move("images/{}".format(image),"failDetectIMG")
                cv2.destroyAllWindows()

            elif cv2.waitKey(0) == ord('d'): #gürültülü resimler
                shutil.move("images/{}".format(image),"noise_IMG")
                cv2.destroyAllWindows()
        else:
            x_min=ObjectsList[0][1]
            y_min=ObjectsList[0][0]
            x_max=ObjectsList[0][3]
            y_max=ObjectsList[0][2]
        
        
        
            cv2.imshow(image, r_image)
            if cv2.waitKey(0) == ord('d'): #gürültülü resimler
                shutil.move("images/{}".format(image),"noiseIMG")
                cv2.destroyAllWindows()

            if cv2.waitKey(0) == ord('s'): #xml dosyalarını kaydetme
                yolo.xml(x_min,y_min,x_max,y_max,image)
                shutil.move("{}.xml".format(image.split(".")[0]),"xml/{}.xml".format(image.split(".")[0]))
                shutil.move("images/{}".format(image),"success")
                cv2.destroyAllWindows()
            
    yolo.close_session()