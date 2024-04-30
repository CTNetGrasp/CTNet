import pybullet as p
import pybullet_data
import time
import math
import cv2
import os
import numpy as np
import sys

sys.path.append('./') 

from utils.ur5_grasp_vilg import UR5_Environment
from utils.simEnv_ur5 import SimEnv
import random
from utils.camera import Camera
from inference_vis import inference

from copy import deepcopy
FINGER_L1 = 0.015
FINGER_L2 = 0.005

def calcAngle2(angle):
    return angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi

def drawGrasps(img, grasps, mode='line'):
    assert mode in ['line', 'region']

    num = len(grasps)
    for i, grasp in enumerate(grasps):
        row, col, angle, width = grasp

        color_b = 255 / num * i
        color_r = 0
        color_g = -255 / num * i + 255

        if mode == 'line':
            width = width / 2

            angle2 = calcAngle2(angle)
            k = math.tan(angle)

            if k == 0:
                dx = width
                dy = 0
            else:
                dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
                dy = k * dx

            if angle < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 1)

            if angle2 < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 1)

            cv2.circle(img, (col, row), 2, (color_b, color_g, color_r), -1)


        else:
            img[row, col] = [color_b, color_g, color_r]
        #

    return img

def run(database_path): 
    ourmodel = inference()
    ourmodel.infer(np.zeros(shape=(640, 480, 3)), 'test') 

    ur5e_grasp_env = UR5_Environment()  
    env = SimEnv(p, database_path, ur5e_grasp_env.ur5e) 

    camera = Camera()  
    env.loadObjsInURDF()

    languages = env.urdfs_exp

    time.sleep(10)
    success_grasp = 0    
    sum_grasp = 0  
    continue_fail = 0 
    id = 0
    while True:    

        color_D435, depth_D435, segm_D435 = env.render_camera_D435(env.agent_cams[0])   
        color_image = cv2.cvtColor(color_D435, cv2.COLOR_RGB2BGR)  
        cv2.imwrite("./input_image_D435.png",color_image)   
        camera_depth = depth_D435 
  
        text = languages[id]
        print('*' * 100)
        print("Expression:", text)
        print('*' * 100)
        text_id = p.addUserDebugText(text, [-0.2, -0.2, 0.9],textColorRGB=[0, 0, 0], textSize=1.5)
        (col, row), grasp_angle, grasp_width_pixels = ourmodel.infer(color_image, text) 

        row, col = int(row), int(col)
        grasp_angle = np.deg2rad(grasp_angle)


        grasp_width = camera.pixels_TO_length(grasp_width_pixels, camera_depth[row, col]) 
        grasp_x, grasp_y, grasp_z = camera.img2world([col, row], camera_depth[row, col]) 
        
        camera_x = np.array(camera.front_position_d435[0])
        camera_z = np.array(camera.front_position_d435[2])

        x_length = camera_x - grasp_x
        grasp_z = camera_z - np.sqrt( (camera_depth[row, col]**2 - (x_length)**2) )
        
        grasp_z = max(grasp_z - 0.08, 0.002) 


        print('*' * 100)
        print('grasp center in the camera coordinate system')
        print('grasp_x = ', col, 'grasp_y = ', row)

        print('grasp center in the world coordinate system:')
        print('grasp_x = ', grasp_x, 'grasp_y = ', grasp_y)

        print('*' * 100)

 
        im_grasp = drawGrasps(color_image, [[row, col, grasp_angle, grasp_width_pixels]], mode='line')  
        cv2.imwrite("./grasp_image.png",im_grasp)   
        cv2.imshow('im_grasp', im_grasp)
        cv2.waitKey(30)

 
        n = 1  
        for i in range(n):
            p.stepSimulation()
            
            done = ur5e_grasp_env.step1_grasp([grasp_x, grasp_y, grasp_z], grasp_angle, grasp_width/2) 

        
        sum_grasp += 1
        if env.evalGraspAndRemove(z_thresh=0.2, obj_id=id):
            success_grasp += 1

        else:
            continue_fail += 1

        print(f"success_num:{success_grasp}, fails_num: {continue_fail}")
        p.removeUserDebugItem(text_id)
        time.sleep(1)
        ur5e_grasp_env.go_home()
        id += 1
        
        if env.num_urdf == 0:
            # p.stopStateLogging(log_id)
            return success_grasp, sum_grasp



if __name__ == "__main__":

    database_path = 'scenerios\\scene1.txt'

    success_grasp, all_grasp = run(database_path)
    print('\n>>>>>>>>>>>>>>>>>>>> Success Rate: {}/{}={}'.format(success_grasp, all_grasp, success_grasp/all_grasp))     




