import serial
import time
import cv2 #record
import threading #while record do anyother thing
import sys
from micropyGPS import MicropyGPS #NMEA decord
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import json  
import time
import os

ser = serial.Serial("/dev/ttyUSB1", baudrate=115200,timeout=100.0)#Lidar
ser2 = serial.Serial("/dev/ttyUSB0", baudrate=9600)#GPS

def take_pic():
    cap = cv2.VideoCapture(0)
    result, image = cap.read()
    vid_cod = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter('/home/pi/Desktop/violation_video.avi', vid_cod, 40.0, (640,480))
    while cap.isOpened():
        cv2.imwrite("/home/pi/Desktop/violaiton_photo.jpeg", image)
        start=time.time()
        end=0
        while end-start<10:
            ret,frame=cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            output.write(frame)
            end=time.time()
        cap.release()
        output.release() 
        print("Finish cam")

def getTFminiData():
    global distance
    while 1:
        time.sleep(1)
        count = ser.in_waiting
        if count > 15:
            recv = ser.read(9)   
            ser.reset_input_buffer() 
            if recv[0] == 0x59 and recv[1] == 0x59:    
                distance = recv[2] + recv[3] * 256
                strength = recv[4] + recv[5] * 256
                ser.reset_input_buffer()
                #print(distance)

def t1_run():
    while 1:
        getTFminiData()
        #gpscount()
        #print(1)

def gpscount():
    mgps=MicropyGPS()
    global lan,lon,speed
    while True:
        #if ser2.in_waiting: 
            #data_raw = ser2.readline()
            #msen=data_raw
        msen = '$GPRMC,055148,A,2407.8945,N,12041.7649,E,000.0,000.0,061196,003.1,W*69'
        for x in msen:
            mgps.update(x)
        lan=mgps.latitude[0]+mgps.latitude[1]/100
        lon=mgps.longitude[0]+mgps.longitude[1]/100
        speed=mgps.speed[2]
        #print(lan,lon,speed)
        return speed

def send():
    t = time.localtime()
    result = time.strftime("%Y-%m-%d,%H-%M-%S", t)
    f=open("/home/pi/Desktop/violaiton_photo.jpeg", "rb") #3.7kiB in same folder
    fileContent = f.read()
    byteArr = bytearray(fileContent)
    client.publish("user/car/photo/"+result+"/"+str(lan)+"/"+str(lon), byteArr)

    v=open("/home/pi/Desktop/violation_video.avi", "rb") #3.7kiB in same folder
    #print(os.path.exists("/home/pi/Desktop/violation_video.avi"))
    fileContent = v.read()
    byteArr1 = bytearray(fileContent)
    client.publish("user/violation_videos/"+result+"/"+str(lan)+"/"+str(lon),byteArr1).wait_for_publish(timeout=None)
    print('Finish Send')

def t2_run():
    send()

if __name__ == '__main__':
    t1=threading.Thread(target=t1_run)
    #t2=threading.Thread(target=t2_run)
    gpscount()
    distance=0
    speed=45
    safe_dis=20
    client = mqtt.Client()
    client.username_pw_set("iot","isuCSIE2022#")
    client.connect("140.127.196.119", 18315, 60)
    client.loop_start()
    cnt=0
    is_send=False
    is_take=False
    try:
        if ser.is_open == False:
            ser.open()
        if ser2.is_open ==False:
            ser2.open()
        t1.start()
        while 1:
            if distance<safe_dis and speed>30:
                print("Dangerous")
                if is_take==False:
                    print('Start Recording')
                    print("wait 8 sec")
                    take_pic()
                    is_take=True
                    cnt+= 8
                else:
                    cnt+=0.5
                    time.sleep(0.5)
                    print(cnt,"sec")
                    if cnt>12  and is_send==False:
                        print("Violation")
                        send()
                        is_send=True
            else:
                cnt=0
                is_send=False
                is_take=False
                time.sleep(1)

    except KeyboardInterrupt:   # Ctrl+C
        if ser != None:
            ser.close()
        if ser2 != None:
            ser2.close()