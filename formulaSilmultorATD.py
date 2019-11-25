import gym
import gym
from PIL import Image
import numpy as np
from silmulator import CarRacing
import matplotlib.pyplot as plt
import cv2


def transform(s):
#     cv2.imshow('original', s)
#     cv2.waitKey(1)

    # crop_img = img[200:400, 100:300] # Crop from x, y, w, h -> 100, 200, 300, 400
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    # bottom_black_bar is the section of the screen with steering, speed, abs and gyro information.
    # we crop off the digits on the right as they are illigible, even for ml.
    # since color is irrelavent, we grayscale it.
    upper_color_bar=s[:840,:840]
    #plt.imshow(upper_color_bar)
    #plt.show()
    bottom_black_bar = s[840:, 120:]
   # plt.imshow(bottom_black_bar)
   # plt.show()
    img = cv2.cvtColor(bottom_black_bar, cv2.COLOR_RGB2GRAY)
    bottom_black_bar_bw = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
    bottom_black_bar_bw = cv2.resize(bottom_black_bar_bw, (840, 120), interpolation = cv2.INTER_NEAREST)
    img2= cv2.cvtColor(upper_color_bar, cv2.COLOR_RGB2GRAY)
    upper_color_bar_bw=cv2.threshold(img2, 1, 255, cv2.THRESH_BINARY)[1]
    upper_color_bar_bw=cv2.resize(upper_color_bar_bw, (840, 840), interpolation = cv2.INTER_NEAREST)
    
    # upper_field = observation[:84, :96] # this is the section of the screen that contains the track.
    upper_field = s[:84, 6:90] # we crop side of screen as they carry little information
    img = cv2.cvtColor(upper_field, cv2.COLOR_RGB2GRAY)
    upper_field_bw = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]
    upper_field_bw = cv2.resize(upper_field_bw, (10, 10), interpolation = cv2.INTER_NEAREST) # re scaled to 7x7 pixels
#     cv2.imshow('video', upper_field_bw)
#     cv2.waitKey(1)
    upper_field_bw = upper_field_bw.astype('float')/255
        
    car_field = s[660:780, 43:53]
    img = cv2.cvtColor(car_field, cv2.COLOR_RGB2GRAY)
    car_field_bw = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)[1]

#     print(car_field_bw[:, 3].mean()/255, car_field_bw[:, 4].mean()/255, car_field_bw[:, 5].mean()/255, car_field_bw[:, 6].mean()/255)
    car_field_t = [car_field_bw[:, 3].mean()/255, car_field_bw[:, 4].mean()/255, car_field_bw[:, 5].mean()/255, car_field_bw[:, 6].mean()/255]
    
#     rotated_image = rotateImage(car_field_bw, 45)
#     cv2.imshow('video rotated', rotated_image)
#     cv2.waitKey(1)
   # plt.imshow(upper_field_bw)
    #plt.imshow()
    #plt.show()
    return bottom_black_bar_bw , upper_field_bw, car_field_t,upper_color_bar_bw


def compute_steering_speed_gyro_abs(a,count):
    #b=a.rechape
    #img = Image.fromarray(a[0][10], 'RGB')
    #img.save('my.png')
    #img.show()   
    #my_a = np.random.normal(0,1,size=[100,100])
    if(count>=200):
        print((a[60, 460:600].mean()/255))
        #plt.imshow(a)
        #plt.show()

    right_steering = a[60, 360:460].mean()/255
    left_steering = a[60, 260:360].mean()/255
    steering = (right_steering - left_steering)/2.0 #(right_steering - left_steering + 1.0)/2
    left_gyro = a[60, 460:600].mean()/255
    right_gyro = a[60, 600:760].mean()/255
    gyro = (right_gyro - left_gyro )/2.0
    speed =a[:, 0][:-2].mean()/255   #a[:, 0][:-2].mean()/255
    abs1 = a[:, 60][:-2].mean()/255
    abs2 = a[:, 80][:-2].mean()/255
    abs3 = a[:, 100][:-2].mean()/255
    abs4 = a[:, 120][:-2].mean()/255    
    white = np.ones((round(int(speed) * 100), 10))
    black = np.zeros((round(100 - int(speed) * 100), 10))
    #speed_display = np.concatenate((black, white))*255    
    #cv2.imshow('sensors', speed_display)
    #cv2.waitKey(1)
    return [steering, speed, gyro, abs1, abs2, abs3, abs4]

class SilumationDetails:
    def __init__(self,num_of_episodes=1,num_of_steps=400):
        self.environment=CarRacing()
        #self.environment=gym.make('CarRacing-v0')        
        self.observation=self.environment.reset()
        self.numOfEpisodes=num_of_episodes
        self.numOfSteps=num_of_steps
        self.reward=0
        self.done=False
        self.information=0
        self.actualSteering=0
        self.actualSpeed=0
        self.steeringDiagram=[]
        self.speedDiagram=[]
        self.gyroDiagram=[]
        self.actualGyro=0
        self.aps=[]
        self.count=0


    def getEnviorment(self):
        return self.environment   


    def DoNextAction(self,action):
        self.environment.render()
        self.observation, self.reward, self.done, self.information = self.environment.step(action)
        bottom_black_bar_bw , upper_field_bw, car_field_t,upper_black_bar_bw =transform(self.observation)
        self.count+=1
        detais=[]
        detais=compute_steering_speed_gyro_abs(bottom_black_bar_bw,self.count)
        self.actualSteering=detais[0]
        self.actualSpeed=detais[1]
        self.actualGyro=detais[2]  
        self.speedDiagram.append(self.actualSpeed)
        self.gyroDiagram.append(self.actualGyro)
        self.steeringDiagram.append(self.actualSteering)
        #return [steering, speed, gyro, abs1, abs2, abs3, abs4]


    def reward(self):
        return self.reward
    
    def GetNumOfEpisodes(self):
        return self.numOfEpisodes


    def GetNumOfSteps(self):
        return self.numOfSteps

    def isDone(self):
        return self.done    


    def GetActualSpeedCar(self):
        return self.actualSpeed


    def GetActalGyroCar(self):
        return self.actualGyro 
    
       
    def ExitSilmulator(self):
        self.environment.close()
           
    def PlotSpeed(self):
        speed=np.array(self.speedDiagram, dtype=None, copy=True, order='K', subok=False, ndmin=0)
        plt.plot(np.arange(self.numOfSteps),speed, 'ro')
        plt.axis([0,self.numOfSteps, 0,max(speed)+2])
        plt.show()
   

    def PlotGyroDiagram(self):
        gyro=np.array(self.gyroDiagram, dtype=None, copy=True, order='K', subok=False, ndmin=0)
        plt.plot(np.arange(self.numOfSteps),gyro, 'ro')
        plt.axis([0,self.numOfSteps, min(gyro),max(gyro)])
        plt.show()

    def PlotSteering(self):
        steering=np.array(self.steeringDiagram, dtype=None, copy=True, order='K', subok=False, ndmin=0)
        plt.plot(np.arange(self.numOfSteps),steering, 'ro')
        plt.axis([0,self.numOfSteps, min(steering)-1,max(steering)+1])
        plt.show()    

    

if __name__ == "__main__":
    mySilmulator=SilumationDetails(100,100)
    
    for i in range(0,mySilmulator.GetNumOfSteps()):
        env=mySilmulator.getEnviorment()
        action = env.action_space.sample() # your agent here (this takes random actions)
      #  mySilmulator.DoNextAction([0,1,0])
        mySilmulator.DoNextAction(action)
       # print(mySilmulator.GetActualSpeedCar())

    mySilmulator.PlotGyroDiagram()

    mySilmulator.ExitSilmulator()
