"""
Copyright {2021} {Robert Arvind}
   Licensed under the BSD 3-Clause License (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       https://opensource.org/licenses/BSD-3-Clause
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
#import argparse 
from tkinter import *
import tkinter as tk
#from PIL import Image
#from PIL import ImageTk
from tkinter import filedialog
from tkinter import simpledialog

app_window = tk.Tk() 

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -10,-10
x_, y_ = -10,-10

#app_window = tk.Tk()    

def dup1(image_path4, temp4, height, width):
    
    img4 = cv2.imread(image_path4)
    #if (height is not None) and (width is not None):
    #    img4 = cv2.resize(img4,(width, height), interpolation = cv2.INTER_CUBIC)
    
    img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    
    rs4 = cv2.matchTemplate(img4,temp4,cv2.TM_CCOEFF_NORMED)
    
    
    retval4 = np.nanmax(rs4)
    #retval4 = np.max(rs4)
    
    
    
    return retval4


def createmaskk(output_dir6, toutput_dir6, doutput_dir6, moutput_dir6, image_path6, image_name6, tempp6, ww6, hh6, height, width):
    
    imagy6 = cv2.imread(image_path6)
    mask6 = np.zeros(imagy6.shape[:2],np.uint8)

    bgdModel6 = np.zeros((1,65),np.float64)
    fgdModel6 = np.zeros((1,65),np.float64)
    
    #rawimg6 = np.copy(imagy6)
    #rawimg6 = imagy6.copy()
    #if (height is not None) and (width is not None):
    #    imagy6 = cv2.resize(imagy6, (width, height), interpolation = cv2.INTER_CUBIC)
    
    rawimg6 = np.copy(imagy6)
    #rawimg6 = imagy6.copy()
    
    imagyy6 = cv2.imread(image_path6)
    #if (height is not None) and (width is not None):
    #    imagyy6 = cv2.resize(imagyy6, (width, height), interpolation = cv2.INTER_CUBIC)
    
    lolmask6 = np.zeros(imagy6.shape[:2],np.uint8)
    
    imagy6 = cv2.cvtColor(imagy6,cv2.COLOR_BGR2GRAY)
    imagxy6 = imagy6.copy()
    
    
    rees6 = cv2.matchTemplate(imagy6,tempp6,cv2.TM_CCOEFF_NORMED)
    min_vall6, max_vall6, min_locc6, max_locc6 = cv2.minMaxLoc(rees6)  
    leftop6 = max_locc6  
    temper6 = imagyy6[leftop6[1]:leftop6[1] + hh6, leftop6[0]:leftop6[0] + ww6]
    
    rightbot6 = (leftop6[0] + ww6, leftop6[1] + hh6)
    rect6 = (leftop6[0],leftop6[1],rightbot6[0],rightbot6[1])
    cv2.grabCut(imagyy6,mask6,rect6,bgdModel6,fgdModel6,15,cv2.GC_INIT_WITH_RECT)
    cv2.rectangle(imagxy6,leftop6, rightbot6, (255,255,255), 1)
    #cv2.rectangle(imagxy6,leftop6, rightbot6, (0,0,0), 1)
    
    mask26 = np.where((mask6==2)|(mask6==0),0,255).astype('uint8')
    #mmask6 = np.where((mask6==2)|(mask6==0),0,1).astype('uint8')
    
    for i in range(0,lolmask6.shape[0]):
        for j in range(0,lolmask6.shape[1]):
            if ((i >= leftop6[0]) and (j >= leftop6[1]) and (i <= rightbot6[0]) and (j <= rightbot6[1]) and (mask26[j][i] == 255)):
                lolmask6[j][i] = 255
            
    
    cv2.imwrite(toutput_dir6+'/'+image_name6, temper6)
    cv2.imwrite(doutput_dir6+'/'+image_name6, imagxy6)
    cv2.imwrite(output_dir6+'/'+image_name6, rawimg6)
    cv2.imwrite(moutput_dir6+'/'+image_name6, lolmask6)
    

def draw_rect(event,x,y,flags,param):
    global ix,iy,drawing,mode,x_,y_
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        x_,y_ = x,y
        qset.add((x_,y_))
        
    elif event == cv2.EVENT_MOUSEMOVE:
        pic = imggx.copy()
        x_,y_ = x,y
        if drawing == True:
            if mode == True:
                cv2.rectangle(pic,(ix,iy),(x_,y_),(255,255,255),1) #(0,0,0)
                cv2.imshow('image',pic)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(imggx,(ix,iy),(x,y),(255,255,255),1) #(0,0,0)
            
        pset.add((x,y))
        
        
def imgresize(output, impath, imname, height, width):
    
    
    imgr = cv2.imread(impath)
    
    imgr = cv2.resize(imgr,(width, height), interpolation = cv2.INTER_CUBIC)
    
    cv2.imwrite(output+'/'+imname, imgr)
        

#app_window = tk.Tk()

answer = messagebox.askyesno("Question","Do all training images have the same resolution?")

height = None

width = None

if answer is False:

    height = simpledialog.askinteger("Input Height", "Enter height:", parent=app_window, minvalue=0, maxvalue=100000)

    width = simpledialog.askinteger("Input Width", "Enter width:", parent=app_window, minvalue=0, maxvalue=100000)


if (height is not None) and (width is not None):

    path1 = filedialog.askdirectory()
    detection1 = path1
    directory1 = os.path.dirname(path1)
    
    resized_data = directory1 + "/resized_data"
    
    try:
        os.mkdir(resized_data)
    except OSError:
        crazy = -10
    
    for fln in sorted(os.listdir(detection1)):
    
        imgresize(resized_data, detection1+'/'+fln, fln, height, width)
        
    detection = resized_data
        
        
#elif (height is None) and (width is None):
else:

    path = filedialog.askdirectory()

    detection = path


directory = os.path.dirname(detection)

#app_window.mainloop() 

detectednew = directory + "/detectednew"

try:
    os.mkdir(detectednew)
except OSError:
    funny1 = -10
    

    
templates = directory + "/templates"
    
try:
    os.mkdir(templates)
except OSError:
    funny2 = -10
    
    
bounds = directory + "/bounds"
    
try:
    os.mkdir(bounds)
except OSError:
    funny3 = -10
    
    
trains = directory + "/trains"
    
try:
    os.mkdir(trains)
except OSError:
    funny4 = -10

        
jk = -5.0
joke1 = [jk for i in range(len(sorted(os.listdir(detection))))]

testarr1 = np.array(joke1, dtype=np.float64)


pset = set()
qset = set()

itr = 0

setcheck = set()

for file1 in sorted(os.listdir(detection)):
    
    
    
    if (len(sorted(os.listdir(detectednew))) != 0):
        
        imgg = cv2.imread(detectednew+'/'+file1)
        #if (height is not None) and (width is not None):
        #    imgg = cv2.resize(imgg,(width, height), interpolation = cv2.INTER_CUBIC)
            
        imgg = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
        
            
        iimg = cv2.imread(detection+'/'+file1)     
        #if (height is not None) and (width is not None):
        #    iimg = cv2.resize(iimg,(width, height), interpolation = cv2.INTER_CUBIC)
            
        iimg = cv2.cvtColor(iimg, cv2.COLOR_BGR2GRAY)
        
        imgg1 = np.copy(iimg)
        
        
    else:
        
        imgg = cv2.imread(detection+'/'+file1)
        
        
        #if (height is not None) and (width is not None):
        #    imgg = cv2.resize(imgg,(width, height), interpolation = cv2.INTER_CUBIC)
            
        imgg = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
        
        imgg1 = np.copy(imgg)
    
    
    imggx = np.copy(imgg)
    
    
    smg = np.copy(imggx)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_rect)
    
    
    while(1):
    
        
        cv2.imshow('image',imggx)
        
        if not cv2.EVENT_MOUSEMOVE:
            pic = imggx.copy()
            if mode == True:
                
                cv2.imshow('image',pic)
                
        k = cv2.waitKey(0) & 0xFF
        
        if k == ord('y'):
            
            break
            
        elif k == ord('n'):
            
            pset.clear()
            qset.clear()
            x_,y_ = -10,-10
            ix,iy = -10,-10
            
            imggx = np.copy(smg)
            
            
    cv2.destroyAllWindows()
    
    
    if pset and qset:
        
        pt = pset.pop()
        qt = qset.pop()
       
        
        px = pt[0]
        py = pt[1]
        
        qx = qt[0]
        qy = qt[1]
        
        if (px >= qx) and (py >= qy):
        
            temp = imgg1[qy:qy+abs(py-qy), qx:qx+abs(px-qx)]
            
        elif (px < qx) and (py < qy):
        
            temp = imgg1[py:py+abs(py-qy), px:px+abs(px-qx)]
            
        elif (px < qx) and (py >= qy):
        
            temp = imgg1[qy:qy+abs(py-qy), px:px+abs(px-qx)]
            
        elif (px >= qx) and (py < qy):
        
            temp = imgg1[py:py+abs(py-qy), qx:qx+abs(px-qx)]
        
            
        w, h = temp.shape[::-1]
        
    else:
        
        temp = cv2.imread(templates+'/'+file1)
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        w, h = temp.shape[::-1]
    
    pset.clear()
    qset.clear()
    
    cnt = 0
    
    
    for file2 in sorted(os.listdir(detection)):
        
        if file2 not in setcheck:
            
            res = dup1(detection+'/'+file2, temp, height, width)
        
            if res >= testarr1[cnt]: 
                
                testarr1[cnt] = res
                
                createmaskk(trains, templates, detectednew, bounds, detection+'/'+file2, file2, temp, w, h, height, width)
                
            
        cnt = cnt + 1
        
    setcheck.add(file1)
        
    
    itr = itr + 1
        
app_window.mainloop()       

