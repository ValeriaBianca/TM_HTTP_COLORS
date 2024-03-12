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

scene = r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\TM_HTTP_2024\mosaic_ref\Scene\Scene1.jpg"
sceneim = cv2.imread(scene)
#cv2.imshow("sceneim",sceneim)

#DISCLAIMER: IMREAD ASSUMES BGR!!!!!!!!!!!!!

#-- Color Thresholds RGB notation || Sidenote: (HSV opencv: 0-179; 0-255; 0-255 || HSV standard paint: 0-360; %; %, ho usato una semplice proporzione per convertire in HSV opencv)
#Check: mask, Thresholding
#-- Azzurro VV
Light_A = np.array([222,231,255], np.uint8)
Dark_A = np.array([92, 139, 94], np.uint8)

#-- Bianco VX
Light_B = np.array([198,28,134], np.uint8)
Dark_B = np.array([99,0,24], np.uint8)

#Light_B2 = np.array([255,43,118], np.uint8)
#Dark_B2 = np.array([130,0,36], np.uint8)

#Light_B = cv2.cvtColor(Light_B, cv2.COLOR_RGB2HSV)
#Dark_B =cv2.cvtColor(Dark_B, cv2.COLOR_RGB2HSV)

#-- Grigio
Light_G = np.array([134,49,108], np.uint8)
Dark_G =np.array([50,3,59], np.uint8)

#Light_G = cv2.cvtColor(Light_G, cv2.COLOR_RGB2HSV)
#Dark_G =cv2.cvtColor(Dark_G, cv2.COLOR_RGB2HSV)

#-- Nero VV
Light_N = np.array([17,194,194], np.uint8) 
Dark_N = np.array([0,0,0], np.uint8)

#Light_N = cv2.cvtColor(Light_N, cv2.COLOR_RGB2HSV)
#Dark_N =cv2.cvtColor(Dark_N, cv2.COLOR_RGB2HSV)

#-- Turchese VV
Light_T = np.array([255,255,92], np.uint8)
Dark_T = np.array([43,75,71], np.uint8)

#Light_T = cv2.cvtColor(Light_T1, cv2.COLOR_RGB2HSV)
#Dark_T =cv2.cvtColor(Dark_T1, cv2.COLOR_RGB2HSV)

#-- Verde V
Light_V = np.array([255,255,66], np.uint8)
Dark_V = np.array([42,47,38], np.uint8)

#Light_V = cv2.cvtColor(Light_V1, cv2.COLOR_RGB2BGR)
#Dark_V =cv2.cvtColor(Dark_V1, cv2.COLOR_RGB2BGR)

#-- Color recognition

#a) Picking frame and converting it to HSV
sceneim_hsv = cv2.cvtColor(sceneim, cv2.COLOR_BGR2HSV)
#cv2.imshow("scenehsv",sceneim_hsv)

#b) Create mask for every color with dark and light value
Mask_Azzurro = cv2.inRange(sceneim_hsv, Dark_A, Light_A)

#B1 = cv2.inRange(sceneim_hsv, Dark_B, Light_B)
#B2 = cv2.inRange(sceneim_hsv, Dark_B2, Light_B2) 
#Mask_Bianco = B1+B2
Mask_Bianco = cv2.inRange(sceneim_hsv, Dark_B,Light_B)
Mask_Grigio = cv2.inRange(sceneim_hsv, Dark_G, Light_G) 
Mask_Nero = cv2.inRange(sceneim_hsv, Dark_N, Light_N)
Mask_Turchese = cv2.inRange(sceneim_hsv, Dark_T, Light_T)
Mask_Verde = cv2.inRange(sceneim_hsv, Dark_V, Light_V)

#c) Threshold
Thresh = 100
Thresh_A = 100  # to tune
# threshold e maschera azzurro ok
ret, Thresh_Azzurro = cv2.threshold(Mask_Azzurro,Thresh_A, 255, cv2.THRESH_BINARY+cv2.THRESH_TOZERO)
#Thresh_Azzurro_res = cv2.resize(Thresh_Azzurro,(800,626))
#cv2.imshow("thresh azzurro",Thresh_Azzurro_res)
Thresh_B = 127
Thresh_Bianco = cv2.adaptiveThreshold(Mask_Bianco, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                           cv2.THRESH_BINARY_INV,11,2)
Thresh_Bianco_res = cv2.resize(Thresh_Bianco,(800,626))
#cv2.imshow("thresh bianco",Thresh_Bianco_res)

Thresh_G = 100
ret, Thresh_Grigio = cv2.threshold(Mask_Grigio,Thresh_G, 255, cv2.THRESH_BINARY+cv2.THRESH_TOZERO)
print(ret)
#Thresh_Grigio_res = cv2.resize(Thresh_Grigio,(800,626))
#cv2.imshow("thresh Grigio",Thresh_Grigio_res)

ret, Thresh_Nero = cv2.threshold(Mask_Nero,Thresh, 255, cv2.THRESH_BINARY+cv2.THRESH_TOZERO)
#Thresh_Nero_res = cv2.resize(Thresh_Nero,(800,626))
#cv2.imshow("thresh Nero",Thresh_Nero_res)

Thresh_T = 100
ret, Thresh_Turchese = cv2.threshold(Mask_Turchese,Thresh, 255, cv2.THRESH_BINARY+cv2.THRESH_TOZERO)
#Thresh_Turchese_res = cv2.resize(Thresh_Turchese,(800,626))
#cv2.imshow("thresh Turchese",Thresh_Turchese_res)

Thresh_V=80
Thresh_Verde = cv2.adaptiveThreshold(Mask_Verde, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
Thresh_Verde_res = cv2.resize(Thresh_Verde,(800,626))
cv2.imshow("thresh Verde",Thresh_Verde_res)


#d) Find contour of objects to detect
Cont_azzurro = cv2.findContours(Thresh_Azzurro, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
Cont_azzurro = imutils.grab_contours(Cont_azzurro)

Cont_bianco = cv2.findContours(Thresh_Bianco, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
Cont_bianco = imutils.grab_contours(Cont_bianco)

Cont_grigio = cv2.findContours(Thresh_Grigio, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
Cont_grigio = imutils.grab_contours(Cont_grigio)

Cont_nero = cv2.findContours(Thresh_Nero, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
Cont_nero = imutils.grab_contours(Cont_nero)

Cont_turchese = cv2.findContours(Thresh_Turchese, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
Cont_turchese = imutils.grab_contours(Cont_turchese)

Cont_verde = cv2.findContours(Thresh_Verde, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
Cont_verde = imutils.grab_contours(Cont_verde)


#e) Draw contours
#sceneim=cv2.cvtColor(sceneim, cv2.COLOR_BGR2RGB)
for c in Cont_azzurro:
    Area1 = cv2.contourArea(c)
    if Area1>20000:
        cv2.drawContours(sceneim, [c], -1, (0,255,0), 2)
        M1 = cv2.moments(c)
        x1 = int(M1["m10"]/M1["m00"])
        y1 = int(M1["m01"]/M1["m00"])
        cv2.circle(sceneim, (x1,y1), 5, (255,255,255),-1)
        cv2.putText(sceneim,"Azzurro", (x1-20,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        #break # quando trova il primo contour utile, esce e passa al prossimo colore

for c in Cont_bianco:
    Area2 = cv2.contourArea(c)
    if Area2>20000 and Area2<30000:
        cv2.drawContours(sceneim, [c], -1, (0,255,0), 2)
        M2 = cv2.moments(c)
        x2 = int(M2["m10"]/M2["m00"])
        y2 = int(M2["m01"]/M2["m00"])
        cv2.circle(sceneim, (x2,y2), 5, (255,255,255),-1)
        cv2.putText(sceneim,"Bianco", (x2-20,y2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

for c in Cont_grigio:
    Area3 = cv2.contourArea(c)
    if Area3>4000 :
        cv2.drawContours(sceneim, [c], -1, (0,255,0), 2)
        M3 = cv2.moments(c)
        x3 = int(M3["m10"]/M3["m00"])
        y3 = int(M3["m01"]/M3["m00"])
        cv2.circle(sceneim, (x3,y3), 5, (255,255,255),-1)
        cv2.putText(sceneim,"Grigio", (x3-20,y3), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        #break

for c in Cont_nero:
    Area4 = cv2.contourArea(c)
    if Area4>20000 and Area4<30000:
        cv2.drawContours(sceneim, [c], -1, (0,255,0), 2)
        M4 = cv2.moments(c)
        x4 = int(M4["m10"]/M4["m00"])
        y4 = int(M4["m01"]/M4["m00"])
        cv2.circle(sceneim, (x4,y4), 5, (255,255,255),-1)
        cv2.putText(sceneim,"Nero", (x4-20,y4), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
       #break

for c in Cont_turchese:
    Area5 = cv2.contourArea(c)
    if Area5>4000 :
        cv2.drawContours(sceneim, [c], -1, (0,255,0), 2)
        M5 = cv2.moments(c)
        x5 = int(M5["m10"]/M5["m00"])
        y5 = int(M5["m01"]/M5["m00"])
        cv2.circle(sceneim, (x5,y5), 5, (255,255,255),-1)
        cv2.putText(sceneim,"Turchese", (x5-20,y5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        #break

for c in Cont_verde:
    Area6 = cv2.contourArea(c)
    if Area6>4000 :
        cv2.drawContours(sceneim, [c], -1, (0,255,0), 2)
        M6 = cv2.moments(c)
        x6 = int(M6["m10"]/M6["m00"])
        y6 = int(M6["m01"]/M6["m00"])
        cv2.circle(sceneim, (x6,y6), 5, (255,255,255),-1)
        cv2.putText(sceneim,"Verde", (x6-20,y6), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        #break

#-- Show result
sceneim_res = cv2.resize(sceneim,(800, 626))
cv2.imshow("Color Detection", sceneim_res)

while(1):
    if cv2.waitKey(20) & 0xFF ==27:
        break
cv2.destroyAllWindows()    

#- Save output image
sceneim = cv2.cvtColor(sceneim, cv2.COLOR_BGR2RGB)
plt.imsave(os.path.join(r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\TM_HTTP_2024\mosaic_ref\Scene", "output_img.jpg"), sceneim)

#-- Centroid evaluation

#-- Rotation evaluation

#-- Piling up json file
#result = {
 #           "message":"success",
 #           "annotations":[
 #               {
  #                  "box_cx": float(str(cx)),
  #                  "box_cy": float(str(cy)),
   #                 "box_w": float(str(box_w)),
    #                "box_h": float(str(box_h)),
     #               "label": "DX",
      #              "score": float(str(1.000)),
       #             "rotation": float(str(theta))
#
 #               }
  #          ],
   #         "result": "ImageDX" 
    #    }

#table = [["label",result["annotations"][0]["label"]],["box_cx", result["annotations"][0]["box_cx"]],
#            ["box_cy",result["annotations"][0]["box_cy"]],["box_w",result["annotations"][0]["box_w"]],
#            ["box_h",result["annotations"][0]["box_h"]],["rotation", result["annotations"][0]["rotation"]]]
        #print(table)
#title = "label values"          
#with open(r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\TM_HTTP_2024\json.txt", 'a') as f:
 #   f.write('\n')
  #  f.write(str(title))
   # f.write('\n')
    #f.write(tabulate(table))
    #f.close()