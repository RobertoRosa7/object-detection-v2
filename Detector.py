import cv2, time, os, tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(20)

class Detector:
  def __init__(self):
    pass

  def read_classes(self, classes_file_path):
    with open(classes_file_path, 'r') as f:
      self.classes_list = f.read().splitlines()

      # Color List
      self.color_list = np.random.uniform(low=0, high=255, size=(len(self.classes_list), 3))
      print(len(self.classes_list), len(self.color_list))
  
  def download_model(self, model_url):
    file_name = os.path.basename(model_url)
    self.model_name = file_name[:file_name.index('.')]
    self.cache_dir = './pretrained_models'

    os.makedirs(self.cache_dir, exist_ok=True)
    get_file(fname=file_name, origin=model_url, cache_dir=self.cache_dir, cache_subdir='checkpoints', extract=True)

  
  def load_model(self):
    print('Loading Model... ' + self.model_name)
    tf.keras.backend.clear_session()
    self.model = tf.saved_model.load(os.path.join(self.cache_dir, 'checkpoints', self.model_name, 'saved_model'))
    print('Model ' + self.model_name + ' loaded successfully...')
  

  def create_bounding_box(self, image, threshold=0.5):
    input_tensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = self.model(input_tensor)

    bboxes = detections['detection_boxes'][0].numpy()
    class_indexes = detections['detection_classes'][0].numpy().astype(np.int32)
    class_scores = detections['detection_scores'][0].numpy()

    h, w, c = image.shape

    bboxes_idx = tf.image.non_max_suppression(bboxes, class_scores, max_output_size=50, iou_threshold=threshold, score_threshold=threshold)

    if len(bboxes_idx) != 0:
      for i in bboxes_idx:
        bbox = tuple(bboxes[i].tolist())
        class_confidence = round(100* class_scores[i])
        class_index = class_indexes[i]
        class_label_text = self.classes_list[class_index]
        class_color = self.color_list[class_index]
        
        display_text = '{}: {}%'.format(class_label_text, class_confidence)
        ymin, xmin, ymax, xmax = bbox
        xmin, xmax, ymin, ymax = (xmin * w, xmax * w, ymin * h, ymax * h)
        xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=class_color, thickness=1)
        cv2.putText(image, display_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, class_color, 2)
        
        line_width = min(int((xmax - xmin) * 0.2), int((ymax - ymin) * 0.2))
        cv2.line(image, (xmin, ymin), (xmin + line_width, ymin), class_color, thickness=5)

    return image


  def predict_image(self, image_path, threshold=0.5):
    image = cv2.imread(image_path)
    image = self.create_bounding_box(image, threshold)

    cv2.imwrite(self.model_name + '.jpg', image)
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
