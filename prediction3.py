from PIL import Image
import base64
import flask
from flask_cors import CORS
from flask import request, jsonify
import cv2
import requests
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import easyocr
import os
import matplotlib as plt
UPLOAD_FOLDER = './images'




def prepare_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = "cpu" # we use a CPU Detectron copy
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES =2
    cfg.MODEL.WEIGHTS = 'model_final.pth' # Set path model .pth
    predictor = DefaultPredictor(cfg)

    return(predictor)
    
 
def extraction(image,outputs_detection): 
    
    boxes =outputs_detection.pred_boxes.tensor.numpy()   
    results={}
    i=0
    #extract the regions of interest
    classes =outputs_detection.pred_classes.numpy() 

    for box in boxes:
        
        ROI = Image.fromarray(image).crop([ int(box[0]),int(box[1]),int(box[2]),int(box[3])])

        plt.pyplot.axis('off')

        plt.pyplot.imshow(ROI)
        reader = easyocr.Reader(['fr']) 
        data = np.asarray(ROI)
        text = reader.readtext(data,detail=0,paragraph=True)
        results[str(classes[i])] = text
        i=i+1
    
 
    return results
  
  
  
app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)
predictor = prepare_predictor()
  
@app.route("/predict", methods=["POST"])
def process_request():
    try:
        file1 = request.files['image']
        print('filename'+file1.filename)
        print('upload'+app.config['UPLOAD_FOLDER'])
        path = os.path.join('.\images',file1.filename)
        print('path'+path)
        file1.save(path)
        image = cv2.imread(path) 

    except Exception as e:
        raise e
    outputs=prepare_predictor()(image)
    
    instances = outputs["instances"].to("cpu")
    
    
    #response = extraction(image,instances)
    response={'0': "notre tier",
          '1': "120.000",
          '2': "30.000"}
    for key, value in response.items():
        print(key, ' : ', value)
    return jsonify(response)

app.run(host="localhost", port=3000)



  