#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 08:29:34 2021

@author: adelino
"""

import pathlib
import sys
import cv2
import numpy as np
import os, shutil
from imutils.object_detection import non_max_suppression
import face_recognition
import subprocess

# sys.exit("MODO DE DEPURAÇÃO: Fim do script") 
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
# os.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp", 1);
# --- VARIAVEIS --------------------------------------------------------------
FIND_PEOPLE = True
FIND_FACES = True
# --- CARREGA RELATORIOS -----------------------------------------------------
REPORT_INPUT_DIR = './REPORTS/'
tupleReportTypes = {'*.txt'}
listReportFilename = []
for files in tupleReportTypes:
    listReportFilename.extend(pathlib.Path(REPORT_INPUT_DIR).glob(files))
listReportFilename.sort()

# --- CARREGA VIDEOS ---------------------------------------------------------
VIDEO_INPUT_DIR = './VIDEOS'
tupleVideoTypes = {'*.avi', '*.mp4'}
listVideoFilename = []
for files in tupleVideoTypes:
    listVideoFilename.extend(pathlib.Path(VIDEO_INPUT_DIR).glob(files))
listVideoFilename.sort()
# --- CARRGA LISTA DE FACES --------------------------------------------------
IMAGE_INPUT_DIR = './FACES'
tupleImageTypes = {'*.png', '*.jpg','*.jpeg'}
listImageFilename = []
for files in tupleImageTypes:
    listImageFilename.extend(pathlib.Path(IMAGE_INPUT_DIR).glob(files))
# ----------------------------------------------------------------------------    
TEMP_OUTPUT_DIR = './.temp/'
if not os.path.exists(TEMP_OUTPUT_DIR):
    try:
        os.makedirs(TEMP_OUTPUT_DIR)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
# ----------------------------------------------------------------------------    
VIDEO_OUTPUT_DIR = './VIDEO_OUT/'
# --- CV2 --------------------------------------------------------------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

if (FIND_FACES):
    # ----------------------------------------------------------------------------
    known_face_encodings = []
    known_face_names = []
    known_face_files = []
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
for idx, file in enumerate(listReportFilename):
    file = open(file)
    # read the file as a list
    print('Iniciado arquivo {:s}'.format(file.name))
    video_file_name = file.name.split('/')[1][7:-10]
    
    VideoOutFileNameFullName = VIDEO_OUTPUT_DIR + video_file_name + "_sel.avi"
    if os.path.isfile(VideoOutFileNameFullName):
        print('\tArquivo já processado...')
        continue
    
    idx_video = [i for i,x in enumerate(listVideoFilename) if x.name[0:-4]==video_file_name] 

    idx_video = idx_video[0]
    data = file.readlines()
    file.close()
    
    # sys.exit("MODO DE DEPURAÇÃO: Fim do script")    
    videFileName = listVideoFilename[idx_video]
    cap = cv2.VideoCapture(videFileName.as_posix())
    if (cap.isOpened()== False): 
        runReportStr = "Error opening video stream or file %s.\n" % listVideoFilename[idx_video]
        print(runReportStr)
        continue
    
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 10);
    for line in data:        
        values = line.split('\t')
        if ((len(values) == 1) and FIND_PEOPLE):
            frameData = values[0].split(' ')
            frame_no = int(frameData[1])
            setResult = cap.set(cv2.CAP_PROP_POS_FRAMES,frame_no)
            ret, image = cap.read()
            (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
        		padding=(4, 4), scale=1.1)
            if (np.max(weights) > 0.90):
                ImageName = TEMP_OUTPUT_DIR + videFileName.stem + "_%07i.jpg" % frame_no;
                rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
                pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
                for (xA, yA, xB, yB) in pick:
                    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
                    cv2.imwrite(ImageName,image)
        
        # sys.exit("MODO DE DEPURAÇÃO: Fim do script")
        if ((len(values) == 2) and FIND_FACES):
            frameData = values[0].split(' ')
            frameFace = values[1].split(' ')
            if (not (frameFace[1] == "Unknown\n")):
                frame_no = int(frameData[1])
                setResult = cap.set(cv2.CAP_PROP_POS_FRAMES,frame_no)
                if (not setResult):
                    continue
                ret, image = cap.read()
                rgb_small_frame = image[:, :, ::-1]      
                face_locations = face_recognition.face_locations(rgb_small_frame)
                if (len(face_locations) < 1):
                    continue
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)            
                if (len(face_encodings) < 1):
                    continue
                face_names = []
                ImageName = TEMP_OUTPUT_DIR + videFileName.stem + "_%07i.jpg" % frame_no;
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                    face_names.append(name)

                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                cv2.imwrite(ImageName,image)
            # sys.exit("MODO DE DEPURAÇÃO: Fim do script")
    
    # listImageFiles = []
    # for files in tupleImageTypes:
    #     listImageFiles.extend(pathlib.Path(TEMP_OUTPUT_DIR).glob(files))    
    # listImageFiles.sort()
    
    cmdString = 'ls ./.temp/ -tr -1 | xargs -i echo "file \'./.temp/{}\'" > FLIST.TXT'
    resultSubProcess =  subprocess.Popen(cmdString, shell=True, stdout=subprocess.PIPE).wait()
    VideoOutFileName = video_file_name + "_sel.avi"
    cmdString = 'ffmpeg -v 10 -f concat -safe 0 -i ./FLIST.TXT -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {:s}{:s}'.format(VIDEO_OUTPUT_DIR,VideoOutFileName)
    resultSubProcess =  subprocess.Popen(cmdString, shell=True, stdout=subprocess.PIPE).wait()
    os.remove("FLIST.TXT")
    cmdString = 'rm -rf .temp/'
    resultSubProcess =  subprocess.Popen(cmdString, shell=True, stdout=subprocess.PIPE).wait()
    cmdString = 'mkdir .temp'
    resultSubProcess =  subprocess.Popen(cmdString, shell=True, stdout=subprocess.PIPE).wait()
    print('Finalizado arquivo {:s}'.format(file.name))