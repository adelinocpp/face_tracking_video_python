#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 12:18:24 2020

@author: adelino
"""

# import sys

# from imageai.Detection import ObjectDetection
# from os import path
import face_recognition
import sys
import cv2
# import subprocess as sp
import os
import numpy as np
# import imutils
import pathlib
import math
# from imutils.object_detection import non_max_suppression
# import matplotlib.pyplot as plt
#from non_max_suppression_fast import non_max_suppression_fast

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
# os.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp", 1);

# --- VARIAVEIS --------------------------------------------------------------
MODO_DEPURACAO = False
VIDEO_INPUT_DIR = './VIDEOS'
# tupleVideoTypes = {'*.avi'} 
tupleVideoTypes = {'*.avi', '*.mp4'}
listVideoFilename = []
for files in tupleVideoTypes:
    listVideoFilename.extend(pathlib.Path(VIDEO_INPUT_DIR).glob(files))
listVideoFilename.sort()  

IMAGE_INPUT_DIR = './FACES'
tupleImageTypes = {'*.png', '*.jpg','*.jpeg'}
listImageFilename = []
for files in tupleImageTypes:
    listImageFilename.extend(pathlib.Path(IMAGE_INPUT_DIR).glob(files))
  
listImageFilename.sort()
REPORT_OUTPUT_DIR = './REPORTS/'
TEMP_OUTPUT_DIR = './.temp/'
if not os.path.exists(REPORT_OUTPUT_DIR):
    try:
        os.makedirs(REPORT_OUTPUT_DIR)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
if not os.path.exists(TEMP_OUTPUT_DIR):
    try:
        os.makedirs(TEMP_OUTPUT_DIR)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
# --- CV2 --------------------------------------------------------------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# ----------------------------------------------------------------------------
BOOL_DO_TRACKING_PEOPLE = True
BOOL_DO_TRACKING_FACE   = True
# Salto de segundos para busca. Util para vídeos longos
divSecond = 0;

# sys.exit("MODO DE DEPURAÇÃO: Fim do script")    

runReportFile = './Relatorio_01.txt'
fRunReport = open(runReportFile, 'wt')
    
# --- APENAS IDENTIFICAR PESSOAS ---------------------------------------------
if (BOOL_DO_TRACKING_PEOPLE and (not BOOL_DO_TRACKING_FACE)):
    for idx, FileName in enumerate(listVideoFilename):
        reportFile = ('Report_' + FileName.stem + '_%05d.txt') % idx
        if os.path.isfile(REPORT_OUTPUT_DIR + reportFile):
            continue
        f = open(REPORT_OUTPUT_DIR + reportFile, 'wt') 
        cap = cv2.VideoCapture(FileName.as_posix())
        if (cap.isOpened()== False): 
            runReportStr = "Error opening video stream or file.\n"
            print(runReportStr)
            fRunReport.write(runReportStr)
        if (divSecond > 0):
            frameJump = math.ceil(int(cap.get(cv2.CAP_PROP_FPS))/divSecond)
        else:
            frameJump = 1

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        runReportStr = 'Rodando arquivo %s - TRACKING_PEOPLE %s, TRACKING_FACE %s\n'%(FileName.name,BOOL_DO_TRACKING_PEOPLE,BOOL_DO_TRACKING_FACE)
        print(runReportStr)
        fRunReport.write(runReportStr)
        for i in range(0,length,frameJump):    
            cap.set(cv2.CAP_PROP_POS_FRAMES,i)
            ret, image = cap.read()
            #image = imutils.resize(image, width=min(400, image.shape[1])) 
            (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
        		padding=(4, 4), scale=1.1)
            intPeopleFound = 0        	
            if ( (len(rects) > 0) and (np.max(weights) > 0.7) ):
                NewLine = ('Pessoa %09d\n') %i
                f.write(NewLine)
                intPeopleFound += 1
                if (MODO_DEPURACAO):
                    sys.exit("MODO DE DEPURAÇÃO: Fim do script") 
                    f.close()
                # ImageName = "out/" + FileName.stem + "_%05i.jpg" %i;
                # rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
                # pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
                # for (xA, yA, xB, yB) in pick:
                #     cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
                # cv2.imwrite(ImageName,image)
        runReportStr = 'Arquivo: %s.\n\tEncontrada(s) %d pessoa(s).\n' % (FileName.name,intPeopleFound)
        print(runReportStr)
        fRunReport.write(runReportStr)
        if (MODO_DEPURACAO and (idx ==0)):
            sys.exit("MODO DE DEPURAÇÃO: Fim do script") 
            f.close()
        f.close() 
# ----------------------------------------------------------------------------

known_face_encodings = []
known_face_names = []
known_face_files = []
# --- APENAS IDENTIFICAR PESSOAS E FACES  ------------------------------------
if (BOOL_DO_TRACKING_PEOPLE and BOOL_DO_TRACKING_FACE):    
    # --- CARREGA DADOS DAS FACES --------------------------------------------
    for k, FileName in enumerate(listImageFilename):
        pattern_image = face_recognition.load_image_file(FileName.as_posix())
        if (len(face_recognition.face_encodings(pattern_image)) > 0):
            face_encoding = face_recognition.face_encodings(pattern_image)[0]
            known_face_encodings.append(face_encoding)
            known_face_files.append(FileName.name)
            FaceName = f"Pattern_{format(k,'03d')}"
            if(len(known_face_files[-1].split('_')) > 1): 
                FaceName = known_face_files[-1].split('_')[1]              
            known_face_names.append(FaceName)
    # if (MODO_DEPURACAO):
    #     sys.exit("MODO DE DEPURAÇÃO: Fim do script") 
    # ------------------------------------------------------------------------
    for idx, FileName in enumerate(listVideoFilename):
        reportFile = ('Report_' + FileName.stem + '_%05d.txt') % idx
        if os.path.isfile(REPORT_OUTPUT_DIR + reportFile):
            continue
        f = open(REPORT_OUTPUT_DIR + reportFile, 'w')   
        cap = cv2.VideoCapture(FileName.as_posix())
        if (cap.isOpened()== False): 
            runReportStr =  "Error opening video stream or file.\n"
            print(runReportStr)
            fRunReport.write(runReportStr)
        if (divSecond > 0):
            frameJump = math.ceil(int(cap.get(cv2.CAP_PROP_FPS))/divSecond)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, frameJump);
        else:
            frameJump = 1;
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        intPeopleFound = 0   
        intFaceFound = 0   
        face_names = []
        face_frame = []
        face_index = []
        runReportStr = 'Rodando arquivo %s - TRACKING_PEOPLE %s, TRACKING_FACE %s\n'%(FileName.name,BOOL_DO_TRACKING_PEOPLE,BOOL_DO_TRACKING_FACE)
        print(runReportStr)
        fRunReport.write(runReportStr)
        for i in range(0,length,frameJump):        
            cap.set(cv2.CAP_PROP_POS_FRAMES,i)
            ret, image = cap.read()
            if ((not ret) or (image is None)):
                continue
            imgR, imgC, imgN = image.shape
            if ((imgR < 1) or (imgC < 1) or (imgN < 1)):
                continue
            (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
        		padding=(4, 4), scale=1.1)                	
            if ( (len(rects) > 0) and (np.max(weights) > 0.7) ):
                NewLine = ('Pessoa %09d\n') %i
                f.write(NewLine)
                intPeopleFound += 1
                
            rgb_small_frame = image[:, :, ::-1]        
            face_locations = face_recognition.face_locations(rgb_small_frame)
            if (len(face_locations) < 1):
                continue
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)            
            if (len(face_encodings) < 1):
                continue
                
            for face_encoding in face_encodings:
                face_frame.append(i)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_index.append(face_distances)    
                face_names.append(name)
                NewLine = ('Pessoa %09d\tNome %s\n') %(i,name)
                for j, data in enumerate(face_distances):
                    line = '\tIDX: %d\tMatch: %s\tDistance: %6.5f\tNome: %s\n' %(j,matches[j],data,known_face_files[j])
                    NewLine += line
                f.write(NewLine)
                intFaceFound += 1
                if (MODO_DEPURACAO and (idx ==0)):
                    sys.exit("MODO DE DEPURAÇÃO: Fim do script") 
                    f.close()
            
        runReportStr = 'Arquivo: %s.\n\tEncontrada(s) %d pessoa(s).\n\tFaces %d\n' % (FileName.name,intPeopleFound,intFaceFound)
        print(runReportStr)
        fRunReport.write(runReportStr)
        if (MODO_DEPURACAO and (idx ==0)):
            sys.exit("MODO DE DEPURAÇÃO: Fim do script") 
            f.close()
        f.close() 
        
fRunReport.close()
# ----------------------------------------------------------------------------
# try:
#     os.remove(TEMP_OUTPUT_DIR)
# except OSError as e:
#     print("Error: %s : %s" % (TEMP_OUTPUT_DIR, e.strerror))
# ----------------------------------------------------------------------------
