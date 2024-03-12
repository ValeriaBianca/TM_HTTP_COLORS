from flask import Flask, jsonify, g, Response, request, flash, redirect, url_for, send_from_directory, render_template
from werkzeug.exceptions import HTTPException
from waitress import serve
from PIL import Image
from werkzeug.utils import secure_filename
from matplotlib import pyplot as plt
from tabulate import tabulate

import os
import io
import cv2
import imutils
import numpy as np
import datetime
import time
import socket
import sys
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#-- Addresses -------------------------------------------------------------------------------------------------------
STEREO_MAP = r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\stereoMap.xml"

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

MAIN_FOLDER=r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\TM_HTTP_2024"
UPLOAD_FOLDER = r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\TM_HTTP_2024\mosaic_ref\upload"
#-- !!Far scrivere a matlab nel txt qual'è la prossima task (una stringa tra {Azzurro, Bianco, Grigio, Nero, Turchese, Verde})!!
TASK_ADDRESS = r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\TM_HTTP_2024\NextTask.txt"

INDEX = r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\templates\index.html"
TEMPLATE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(INDEX)))
TEMPLATE_DIR = os.path.join(TEMPLATE_DIR, 'templates')

#-- Colors-------------------------------------------------------------------------------------------------------------
# Creo dizionario di dizionari con maschere di tutti i possibili colori da riconoscere 
# N.B.: colori in BGR
Colors= {
    #-- Azzurro 
    "Azzurro" :{
        "Light" : (127,187,239),
        "Dark" : (99,43,180)
        },
    #-- Grigio
    "Grigio":{
        "Light" : (137,80,130),
        "Dark" : (66,0,82)
        },
    #-- Giallo
    "Giallo":{
         "Light": (83,255,215),
         "Dark": (14,47,154)
    },
    #-- Nero 
    "Nero":{
        "Light" : (179,177,80), 
        "Dark" : (17,17,16)
        },
    #-- Rosso
    "Rosso":{
         "Light": (189,255,255), 
         "Dark": (168,99,0) 
    },

    #-- Turchese 
    "Turchese":{
        "Light" : (92,255,255),
        "Dark" : (71,75,43)
        },

    }

#----------------------------------------------------------------------------------------------------------------------
#-- Undistort and rectify images before giving them to the ORB algorithm to process
#-- Use camera parameters PREVIOUSLY evaluated by stereo calibration script
cv_file = cv2.FileStorage()
cv_file.open(STEREO_MAP, cv2.FILE_STORAGE_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

cv_file.release()

#-- App section-------------------------------------------------------------------------------------------------------
app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

HOST_NAME = 'TM Vision HTTP Server'
HOST_PORT = 80

nu = 0.75

# ========================================================== SYSTEM =================================================================
@app.errorhandler(HTTPException) 
#Handle an exception that did not have an error handler associated with it, or that was raised from an error handler. 
#This always causes a 500 InternalServerError.
def handleException(e):
    '''Return HTTP errors.'''
    TRIMessage(e)
    return e

#@app.errorhandler(400)
#def bad_request(e):
#    return render_template("400.html"), 400

#@app.errorhandler(404)
#def page_not_found(e):
#    return render_template("404.html"), 404

#@app.errorhandler(405)
#def method_not_allowed(e):
#    return render_template("405.html"), 405

#-- Uncomment to have 404,405,400 pages shown if server is opened by browser

#-- Utility functions--------------------------------------------------------------------------------------------------------------------------------------------------------------------
def TRIMessage(message):
    print(f'\n[{datetime.datetime.now(datetime.timezone(datetime.timedelta(0))).astimezone().isoformat(timespec="milliseconds")}] {message}')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def UndistAndRect(img, stereomap_x, stereomap_y):
    frame = cv2.remap(img, stereomap_x, stereomap_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    return frame

def colorPresent(current_task, Colors):
    if current_task!=str("Azzurro") and current_task!=str("Grigio") and current_task!=str("Nero") \
        and current_task!=str("Turchese") and current_task!=str("Giallo") and current_task!=str("Rosso"):
        return False
    elif current_task == '0':
        print("This slot has no color set")
        return False
    else:
        return True
    
def jsonresponse(cx,cy,box_w, box_h, task, theta, message, output):
    #type check
    if type(cx) != int:
        print("cx has to be an integer number")
        cx = int(cx)
    if type(cy) != int:
        print("cy has to be an integer number")
        cy = int(cy)
    if type(box_w) != float:
        print("box_w has to be a float number")
        box_w = float(box_w)
    if type(box_h) != int:
        print("box_h has to be a float number")
        box_h = float(box_h)
    if type(task) != str :
        print("task has to be a string")
        task = str(task)
    if type(theta) != float:
        print("theta has to be a float number")
        theta = float(theta)
    if message != "success" and message != "fail":
        print("message has to be a string containing either success or fail")
        return redirect(request.url)
    if cx == 0 and cy == 0 and box_w == 0 and box_h == 0: #used when the json contains a failure
        result = { 
                    "message": message,
                    "result" : output
                }
        return result
    result = {
            "message": "success", #This message has to be either "success" or "fail"...
            "annotations":[
                {
                    "box_cx": float(str(cx)),
                    "box_cy": float(str(cy)),
                    "box_w": float(str(box_w)),
                    "box_h": float(str(box_h)),
                    "label": str(task),
                    "score": float(str(1.000)),
                    "rotation": float(str(theta))

                }
            ],
            "result": str(output) #"Image if success, None if fail...."
        }
    return jsonify(result)

#-- GET -------------------------------------------------------------------------------------------------------
#by deafult the app.route expects a get request. Here an index page is put for the user to open on the pc side.
@app.route('/') 
def index():
   return render_template('index.html')

@app.route('/api/<string:m_method>', methods=['GET']) 
#dummy GET to try the connection over the TM robot side in the vision node settings
def get(m_method):
    # user defined method
    result = dict()

    if m_method == 'status':
        result = {
            "result": "status",
            "message": "im ok",
            "result": None
        }
        
    else:
        result = {
            "result": "fail",
            "message": "wrong request",
            "result": None
        }
        
    return result

#-- POST -----------------------------------------------------------------------------------------------------------
@app.route('/api/<string:m_method>', methods=['POST'])
def post(m_method):
    #get key/value
    parameters = request.args
    model_id = parameters.get('model_id')
    TRIMessage(f'model_id: {model_id}')
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    #check key/value
    if model_id is None:
        TRIMessage('model_id is not set')
        result = jsonresponse(0,0,0,0,"model_id required",0,"fail","None")
        #result={                    
        #    "message": "fail",
        #    "result": "model_id required"
        #}
        return result
    
    #-- Saving image on pc
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    #-- WARNING: when setting up vision node, name model_id either dx or sx; 
    # Or write a similar if block to the ones below for the specified model_id chosen inside the vision task 

    # DECIDE MODEL_ID PARAMETER AND ACT ACCORDINGLY
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        name = filename.rsplit('.')
        filename = name[0] + "_scene" + "." + name[1]
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('File saved succesfully')

    #-- Image processing---------------------------------------------------------------------------------------------------------------------------------
    
    #-- The query image is picked from the robot camera and corrected
    queryim = cv2.imread(UPLOAD_FOLDER + r"\image_scene.jpg")
    #queryim = UndistAndRect(queryim, stereoMapR_x, stereoMapR_y)
    
    #-- Color recognition algorithm
    #-- Vado a pescare la foto appena fatta e salvata
    scene = r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\TM_HTTP_2024\mosaic_ref\upload\image_scene.jpg"
    sceneim = cv2.imread(scene)
    #DISCLAIMER: IMREAD ASSUMES BGR!!!!!!!!!!!!!
    #--> Far salvare a Matlab un file di testo con il colore da individuare

    #-- Leggo la task da riconoscere
    f = open(TASK_ADDRESS,"r")
    current_task = str(f.read())
    print(str(current_task))

    #-- Check per vedere se il colore è presente o meno nel pool
    if not colorPresent:
        print("Colore non presente nel pool o nomenclatura errata")
        return redirect(request.url)
       
    #-- Color recognition algorithm
    #-- a) Picking frame and converting it to HSV ---------------------------------------------------------------------------------------------------
    sceneim_hsv = cv2.cvtColor(sceneim, cv2.COLOR_BGR2HSV)

    #-- b) Create mask for every color|| I create only the mask I am interested in -------------------------------------------------------------------
    
    #-- Picking lighter and darker shades of the selected color to construct the mask
    Light = Colors[str(current_task)]["Light"]
    Dark = Colors[str(current_task)]["Dark"]
    Mask = cv2.inRange(sceneim_hsv, Dark, Light)
    
    #-- c) Threshold --------------------------------------------------------------------------------------------------------------------------------
    #-- Thresholding alternative 1
    Thresh_Task = cv2.adaptiveThreshold(Mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)

    #-- d) Find contour of objects to detect ---------------------------------------------------------------------------------------------------
    Cont = cv2.findContours(Thresh_Task, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    Cont = imutils.grab_contours(Cont)

    #-- e) Draw contours || Voglio solo il primo contour perchè mi interessa prendere ogni oggetto una volta sola e in ordine, non mi interessa individuarli tutti -------
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # HOWEVER se in un gruppo di tasks ce  ne sono più di una di colore uguale, questo metodo fallirà. In questo caso devo ripensare l'algoritmo in modo che sappia prima 
    # quante task dello stesso colore ci sono ad ogni iterazione
    # Nota: se il robot rimuove ad ogni iterazione un pezzo dal puzzle, questo problema è aggirato 
    # devo fare in modo che faccia una nuova foto solo quando si muove di posizione
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #sceneim=cv2.cvtColor(sceneim, cv2.COLOR_BGR2RGB)
    found = False
    x1 = 0
    y2 = 0
    Area1 = 0
    rotrect = []
    box= []
    theta = 0
    for c in Cont:
        Area1 = cv2.contourArea(c)
        if Area1>20000 and Area1<30000:
            print(Area1)
            cv2.drawContours(sceneim, [c], -1, (0,255,0), 2)
            M1 = cv2.moments(c)
            x1 = int(M1["m10"]/M1["m00"])
            y1 = int(M1["m01"]/M1["m00"])
            cv2.circle(sceneim, (x1,y1), 5, (255,255,255),-1)
            cv2.putText(sceneim,str(current_task), (x1-20,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
            found = True
            rotrect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rotrect)
            box = np.int64(box)
            theta = rotrect[-1]
            # in opencv theta è compreso tra -90 e 0
            if theta < -45:
                theta = -(90 + theta)
            else:
                theta = -theta
            break # quando trova il primo contour utile, esce
        elif Area1>4000 and Area1<=20000:
            print(Area1)
            print("Area is smaller")
            cv2.drawContours(sceneim, [c], -1, (0,255,0), 2)
            M1 = cv2.moments(c)
            x1 = int(M1["m10"]/M1["m00"])
            y1 = int(M1["m01"]/M1["m00"])
            cv2.circle(sceneim, (x1,y1), 5, (255,255,255),-1)
            cv2.putText(sceneim,str(current_task), (x1-20,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
            found = True
            rotrect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rotrect)
            box = np.int64(box)
            theta = rotrect[-1]
            # in opencv theta è compreso tra -90 e 0
            if theta < -45:
                theta = -(90 + theta)
            else:
                theta = -theta
            break
        elif Area1>1000 and Area1<=4000:
            print(Area1)
            print("Area is so small it could very well be wrong")
            #if I am here, instead of breaking at first result, loops for to get a better area 

        #-- iterate loop for next bigger area

    #-- Check per vedere se il colore è stato effettivamente trovato, altrimenti torna in idle state
    if found == False:
        print("No " + str(current_task) + " was found")
        with open(MAIN_FOLDER+'\\json.txt', 'a') as f:
            f.write(tabulate(["Empty"]))
            f.close()
        result = jsonresponse(0,0,0,0,"Color not found",0,"fail","None")
        return result

    #-- Save result
    sceneim_box = sceneim.copy()
    sceneim_box = cv2.drawContours(sceneim_box, [box], 0,(0,0,255),2)
    sceneim_box = cv2.cvtColor(sceneim_box, cv2.COLOR_BGR2RGB)
    sceneim = cv2.cvtColor(sceneim, cv2.COLOR_BGR2RGB)
    plt.imsave(os.path.join(UPLOAD_FOLDER, "output_img_" + str(current_task)) + ".jpg", sceneim)
    plt.imsave(os.path.join(UPLOAD_FOLDER, "output_img_box_" + str(current_task)) + ".jpg", sceneim_box)
    #-- Centroid evaluation done in area loops
    cx = x1
    cy = y1
    #--inserire calcolo di grandezza della scatola e della rotazione (in prima approssimazione considero area del contouring come quella di un quadrato)
    box_w = int(np.ceil(np.sqrt(Area1)))
    box_h = int(np.ceil(np.sqrt(Area1)))

    #-- Rotation evaluation done in loop
    label = str(current_task)

    #-- Piling result in json format to send back to TMFlow
    # Classification [Can be implemented and for this to work the External Classification vision task has to be selected inside the vision job node]
    if m_method == 'CLS':
        result = (0,0,0,0,"No Classification method implemented, yet",0,"fail","None")
        
    # Detection
    elif m_method == 'DET':
        result = jsonresponse(cx,cy,box_w,box_h,current_task,theta,"success","Image")
        # Storing json in txt file but as a table so that matlab can easily read it
        table = [["label",str(current_task)],["box_cx", cx],
            ["box_cy",cy],["box_w",box_w],
            ["box_h",box_h],["rotation", theta]]
        
        title = "label values"            
        with open(MAIN_FOLDER+'\\json.txt', 'a') as f:
            f.write('\n')
            f.write(str(title))
            f.write('\n')
            f.write(tabulate(table))
            f.close()
        print("json sent!")        
            
    # no method
    else:
        #result = {            
        #    "message": "no method",
        #    "result": None            
        #}
        result = jsonresponse(0,0,0,0,"no method",0,"fail","None")   
        with open(MAIN_FOLDER+'\\json.txt', 'a') as f:
            f.write('\n')
            f.write((str(result)))
            f.close()
    
    return result
    

#-- Entry point
if __name__ == '__main__':
    check=False
    try:
        host_addr = ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")] or 
                [[(s.connect(("8.8.8.8", 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) 
        check = True if len(host_addr) > 0 else False
    except Exception as e:
        TRIMessage(e)
    if check == True:
        host_addr = host_addr[-1] if len(host_addr) > 1 else host_addr[0]
        TRIMessage(f'serving on http://{host_addr}:{HOST_PORT}')
    else:
        TRIMessage(f'serving on http://127.0.0.1:{HOST_PORT}')
    serve(app, port=HOST_PORT, ident=HOST_NAME, _quiet=True)