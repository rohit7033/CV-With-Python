import tensorflow as tf
import numpy as np
import cv2
import math
import subprocess
vidcap = cv2.VideoCapture('./../video1.mp4')
success,image = vidcap.read()
count = 0
success = True
fps = vidcap.get(cv2.CAP_PROP_FPS)
total_frames =math.floor( vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print (fps,total_frames)
i = math.floor(fps)

#open file
f = open('output.txt','w')
f.close()


#while success:
k=0
for i in range (1,total_frames):
  success,image = vidcap.read()
  if i%fps==0 :
    k=k+1
    #success,image = vidcap.read()
    image_out_name = './../frame'+str(k)+'.jpg'
    input_to_exe = '--image_file='+image_out_name
    cv2.imwrite(image_out_name,image)     # save frame as JPEG file
    
    with open('output.txt', 'a') as output_f:
      p = subprocess.Popen(['/home/rohit/models/tutorials/image/imagenet/dist/test1/test1',input_to_exe],stdout=output_f,stderr=output_f)
      p.wait()
    #if cv2.waitKey(10) == 27:                     # exit if Escape is hit
    #    break
      output_f.write('\nThis is the \n'+str(k)+'th file for you\n\n')
    output_f.close() 
  if i ==  total_frames :
    #count += 1
    break
  i=i+1
