import cv2
import math
import time
import numpy as np
import scipy.io as scio
import pybullet as p


HEIGHT = 480
WIDTH = 640


def radians_TO_angle(radians):
    """
    弧度转角度
    """
    return 180 * radians / math.pi

def angle_TO_radians(angle):
    return math.pi * angle / 180

def eulerAnglesToRotationMatrix(theta):
    """
    theta: [r, p, y]
    """
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
        
        
                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                    
                    
    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def getTransfMat(offset, rotate):
    mat = np.array([
        [rotate[0, 0], rotate[0, 1], rotate[0, 2], offset[0]], 
        [rotate[1, 0], rotate[1, 1], rotate[1, 2], offset[1]], 
        [rotate[2, 0], rotate[2, 1], rotate[2, 2], offset[2]],
        [0, 0, 0, 1.] 
    ])
    return mat


class Camera:
    def __init__(self):

        self.fov = 60   
        self.length = 0.7   
        self.H = self.length * math.tan(angle_TO_radians(self.fov/2)) 
        self.W = WIDTH * self.H / HEIGHT    

        self.A = (HEIGHT / 2) * self.length / self.H

        self.InMatrix = np.array([[self.A, 0, WIDTH/2 - 0.5], [0, self.A, HEIGHT/2 - 0.5], [0, 0, 1]], dtype=np.float)

        self.front_position_d435 = (0.4, 0, 0.5) 
        front_rotation_d435 = (np.deg2rad(15), np.pi, np.pi / 2)  
        rotMat = eulerAnglesToRotationMatrix(front_rotation_d435)  
        self.transMat = getTransfMat(self.front_position_d435, rotMat)  

    def camera_height(self):
        return self.length
    
    def img2camera(self, pt, dep):

        pt_in_img = np.array([[pt[0]], [pt[1]], [1]], dtype=np.float)
        ret = np.matmul(np.linalg.inv(self.InMatrix), pt_in_img) * dep
        return list(ret.reshape((3,)))

    
    def camera2img(self, coord):

        z = coord[2]
        coord = np.array(coord).reshape((3, 1))
        rc = (np.matmul(self.InMatrix, coord) / z).reshape((3,))

        return list(rc)[:-1]

    def length_TO_pixels(self, l, dep):
        return l * self.A / dep
    
    def pixels_TO_length(self, p, dep):

        return p * dep / self.A
    
    def camera2world(self, coord):

        coord.append(1.)
        coord = np.array(coord).reshape((4, 1))
        coord_new = np.matmul(self.transMat, coord).reshape((4,))
        return list(coord_new)[:-1]
    
    def world2camera(self, coord):

        coord.append(1.)
        coord = np.array(coord).reshape((4, 1))
        coord_new = np.matmul(np.linalg.inv(self.transMat), coord).reshape((4,))
        return list(coord_new)[:-1]

    def world2img(self, coord):

        # 转到相机坐标系
        coord = self.world2camera(coord)
        # 转到图像
        pt = self.camera2img(coord) # [y, x]
        return [int(pt[1]), int(pt[0])]
    
    def img2world(self, pt, dep):

        coordInCamera = self.img2camera(pt, dep)
        return self.camera2world(coordInCamera)


class RealSenseD435: 

    image_size = (480, 640)
    intrinsics = np.array([[462.14, 0, 320], [0, 462.14, 240], [0, 0, 1]])

    front_position = (0.4, 0, 0.5) 
    front_rotation = (np.deg2rad(15), np.pi, np.pi / 2)  
    front_rotation = p.getQuaternionFromEuler(front_rotation)

    left_position = (0, 0.5, 0.75)
    left_rotation = (np.pi / 4.5, np.pi, np.pi / 4)
    left_rotation = p.getQuaternionFromEuler(left_rotation)

    right_position = (0, -0.5, 0.75)
    right_rotation = (np.pi / 4.5, np.pi, 3 * np.pi / 4)
    right_rotation = p.getQuaternionFromEuler(right_rotation)
    
    # Default camera configs. 
    CONFIG = [
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": front_position,
            "rotation": front_rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        },
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": left_position,
            "rotation": left_rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        }, 
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": right_position,
            "rotation": right_rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        },
    ]


if __name__ == '__main__':
    camera = Camera()
    print(camera.InMatrix)


