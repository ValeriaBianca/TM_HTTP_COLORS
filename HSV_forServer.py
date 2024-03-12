import cv2
import numpy as np
import pandas as pd
import os
import io
import sys
import imutils
import colorsys


from matplotlib import pyplot as plt
from tabulate import tabulate
from PIL import Image 

# da mettere nella sezione indirizzi
MAIN_FOLDER=r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\TM_HTTP_2024"
UPLOAD = r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\TM_HTTP_2024\mosaic_ref\upload"
TASK_ADDRESS = r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\TM_HTTP_2024\NextTask.txt"
scene = r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\TM_HTTP_2024\mosaic_ref\Train\all_colors.jpg"

#--> Far salvare a Matlab un file di testo con il colore da individuare

sceneim = cv2.imread(scene)
# Creo dizionario di dizionari con maschere di tutti i possibili colori da riconoscere (QUESTO DIZIONARIO VA MESSO PRIMA DELLA PARTENZA DEL SERVER)
Colors= { # colori in BGR!!!!!!
    #-- Azzurro VV
    "Azzurro" :{
        "Light" : (127,187,239),
        "Dark" : (99,43,180)
        },
    #-- GrigioV
    "Grigio":{
        "Light" : (137,80,130),
        "Dark" : (66,0,82)
        },

    "Giallo":{
         "Light": (83,255,215),
         "Dark": (14,47,154)
    },
    #-- Nero VV
    "Nero":{
        "Light" : (179,177,80), 
        "Dark" : (17,17,16)
        },

    "Rosso":{
         "Light": (189,255,255), 
         "Dark": (168,99,0) 
    },

    #-- Turchese VV
    "Turchese":{
        "Light" : (92,255,255),
        "Dark" : (71,75,43)
        },

    }

#-- Leggo la task da riconoscere
f = open(TASK_ADDRESS,"r")
current_task = str(f.read())
print(str(current_task))

def colorPresent(current_task, Colors):
    if current_task!=str("Azzurro") and current_task!=str("Grigio") and current_task!=str("Nero") \
    and current_task!=str("Turchese") and current_task!=str("Giallo") and current_task!=str("Rosso"):
        return False
    else:
        return True

#-- Check per vedere se il colore è present o meno nel pool
if not colorPresent:
    sys.exit("Colore non presente nel pool o nomenclatura errata")
    

def jsonresponse(cx, cy, box_w, box_h, task, theta):
    #type check
    if type(cx) != float and type(cx) != int:
        print("cx has to be a int")
        sys.exit
    if type(cy) != float and type(cy) != int:
        print("cy has to be an integer number")
        sys.exit
    if type(box_w) != float and type(box_w) != int:
        print("box_w has to be an integer number")
        sys.exit
    if type(box_h) != float and type(box_h) != int:
        print("box_h has to be an integer number")
        sys.exit
    if type(task) != str:
        print("task has to be a string")
        sys.exit
    if type(theta) != float and type(theta) != int:
        print("theta has to be an integer number")
        sys.exit
    result = {
            "message":"success",
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
            "result": "Image" 
        }
    return result


#-- Color recognition
#-- a) Picking frame and converting it to HSV
sceneim_hsv = cv2.cvtColor(sceneim, cv2.COLOR_BGR2HSV)


#-- b) Create mask for every color|| I create only the mask I am interested in
#-- Picking lighter and darker shades of the selected color to construct the mask
Light = Colors[str(current_task)]["Light"]
Dark = Colors[str(current_task)]["Dark"]
Mask = cv2.inRange(sceneim_hsv, Dark, Light)

#-- c) Threshold
Thresh_Task = cv2.adaptiveThreshold(Mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)

#-- d) Find contour of objects to detect
Cont = cv2.findContours(Thresh_Task, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
Cont = imutils.grab_contours(Cont)

#-- e) Draw contours || Voglio solo un contour perchè mi interessa prendere ogni oggetto una volta sola e in ordine, non mi interessa individuarli tutti
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# HOWEVER se in un gruppo di tasks ce ne sono più di una di colore uguale, questo metodo fallirà. 
# In questo caso devo ripensare l'algoritmo in modo che sappia prima 
# quante task dello stesso colore ci sono ad ogni iterazione
# Nota: se il robot rimuove ad ogni iterazione un pezzo dal puzzle, questo problema è aggirato 
# Devo fare in modo che faccia una nuova foto solo quando si muove di posizione
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
found = False #True se trova un'area abbastanza grande e quindi se trova effettivamente il colore in questione 
#-- Inizializzo le variabili per poterle prendere dopo
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

#-- Check per vedere se effettivamente il colore è stato trovato
if found == False:
    with open(MAIN_FOLDER+'\\json.txt', 'a') as f:
            f.write(tabulate(["Empty"]))
            f.close()
    sys.exit(str(current_task) + " assente nella foto.")

#-- Save result
sceneim_box = sceneim.copy()
sceneim_box = cv2.drawContours(sceneim_box, [box], 0,(0,0,255),2)
sceneim_box = cv2.cvtColor(sceneim_box, cv2.COLOR_BGR2RGB)
sceneim = cv2.cvtColor(sceneim, cv2.COLOR_BGR2RGB)
plt.imsave(os.path.join(UPLOAD, "output_img_"+str(current_task))+".jpg", sceneim)
plt.imsave(os.path.join(UPLOAD, "output_img_box_"+str(current_task))+".jpg", sceneim_box)
#-- Centroid evaluation done in area loops
cx = x1
cy = y1
box_w = int(np.ceil(np.sqrt(Area1)))
box_h = int(np.ceil(np.sqrt(Area1)))

#-- Rotation evaluation made inside loop

result = jsonresponse(cx,cy,box_w,box_h,current_task,theta)

table = [["label",result["annotations"][0]["label"]],["box_cx", result["annotations"][0]["box_cx"]],
            ["box_cy",result["annotations"][0]["box_cy"]],["box_w",result["annotations"][0]["box_w"]],
            ["box_h",result["annotations"][0]["box_h"]],["rotation", result["annotations"][0]["rotation"]]]
title = "label values"          
with open(MAIN_FOLDER+'\\json.txt', 'a') as f:
    f.write('\n')
    f.write(str(title))
    f.write('\n')
    f.write(tabulate(table))
    f.close()

