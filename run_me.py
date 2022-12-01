from Detector import *

MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz'
classes_list = "coco.names"
image_path = 'test_images/1.jpg'
threshold = 0.5

detector = Detector()
detector.read_classes(classes_list)

detector.download_model(MODEL_URL)
detector.load_model()
detector.predict_image(image_path, threshold)