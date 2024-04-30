"""
虚拟环境文件
初始化虚拟环境，加载物体，渲染图像，保存图像

(待写) ！！ 保存虚拟环境状态，以便离线抓取测试
"""
from urdf_models import models_data

import pybullet as p
import pybullet_data
import time
import math
import os
import glob
import random
import cv2
import shutil
import numpy as np
import scipy.io as scio
import sys
import scipy.stats as ss
import skimage.transform as skt
# sys.path.append('D:/guyueju/code/pybullet_grasp')
import utils.camera as camera

sys.path.append('./')

# 图像尺寸
IMAGEWIDTH = 640
IMAGEHEIGHT = 480

nearPlane = 0.01
farPlane = 10

fov = 60    # 垂直视场 图像高tan(30) * 0.7 *2 = 0.8082903m 
aspect = IMAGEWIDTH / IMAGEHEIGHT



def imresize(image, size, interp="nearest"):
    skt_interp_map = {
        "nearest": 0,
        "bilinear": 1,
        "biquadratic": 2,
        "bicubic": 3,
        "biquartic": 4,
        "biquintic": 5
    }
    if interp in ("lanczos", "cubic"):
        raise ValueError("'lanczos' and 'cubic'"
                         " interpolation are no longer supported.")
    assert interp in skt_interp_map, ("Interpolation '{}' not"
                                      " supported.".format(interp))

    if isinstance(size, (tuple, list)):
        output_shape = size
    elif isinstance(size, (float)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size
        output_shape = tuple(np_shape.astype(int))
    elif isinstance(size, (int)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size / 100.0
        output_shape = tuple(np_shape.astype(int))
    else:
        raise ValueError("Invalid type for size '{}'.".format(type(size)))

    return skt.resize(image,
                      output_shape,
                      order=skt_interp_map[interp],
                      anti_aliasing=False,
                      mode="constant")

def inpaint(img, missing_value=0):
    """
    Inpaint missing values in depth image.
    :param missing_value: Value to fill in teh depth image.
    """
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale

    return img


class SimEnv(object):
    """
    虚拟环境类
    """
    def __init__(self, bullet_client, path, gripperId=None):
        """
        path: 模型路径 
        """
        self.p = bullet_client 
        
        self.p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, shadowMapIntensity=0, lightPosition=[0, 0, 100]) ## 世界坐标系呈现与否。 
        # self.p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0) ## 世界坐标系呈现与否。 

        self.p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000, solverResidualThreshold=0, enableFileCaching=0) ## 设置物理参数。
        # self.p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22, cameraTargetPosition=[0, 0, 0]) ## 设置初始视角
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 添加搜索路径
        self.models = models_data.model_lib()
        self.planeId = self.p.loadURDF("plane.urdf", [0, 0, 0])  # 加载地面    
        self.workspaceId_zxx = self.p.loadURDF("./myModel/workspace/workspace.urdf" , [0.6, 0, 0] , useFixedBase = True )  # 加载地面 
        p.changeDynamics(
            self.workspaceId_zxx,
            -1, 
            lateralFriction=1.1,
            rollingFriction = 1.1,
            spinningFriction = 1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )

        # self.trayId = self.p.loadURDF('./pybullet_grasp/myModel/tray/tray_small.urdf', [0, 0, -0.007])   # 加载托盘

        self.p.setGravity(0, 0, -9.8) # 设置重力
        self.flags = self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.gripperId = gripperId 

        self.agent_cams = camera.RealSenseD435.CONFIG  

        # 加载相机     
        ## 很重要！需要研究一下; 加载相机时要设置视场fov其实就是角度; aspect设置宽高比;     
        self.movecamera(0.6, 0, 0.7)       
        # self.movecamera_position_rotation(0, 0)      
        self.projectionMatrix = self.p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)     

        # 读取path路径下的list.txt文件 
        list_file = path  ##   
        if not os.path.exists(list_file):  
            raise shutil.Error  
        self.urdfs_list = []  
        self.urdfs_pos = []  
        self.urdfs_ang = []
        self.urdfs_exp = []
        with open(list_file, 'r') as f:   
            while 1:  
                line = f.readline()   
                print(line)
                if not line:    
                    break   
                name, pos, ang, exp = line.split(', ')
                x, y, z = pos.split(' ')
                t1, t2, t3 = ang.split(' ')

                self.urdfs_list.append(self.models[name]) ##  获取模型列表，然后再获取几个参数
                self.urdfs_pos.append([float(x), float(y), float(z)])
                self.urdfs_ang.append([np.deg2rad(float(t1)), np.deg2rad(float(t2)), np.deg2rad(float(t3))])
                self.urdfs_exp.append(str(exp.strip('\n')))

        self.num_urdf = 0
        self.urdfs_id = []  # 存储由pybullet系统生成的模型id
        self.objs_id = []   # 存储模型在文件列表中的索引，注意，只在path为str时有用
        self.EulerRPList = [[0, 0], [math.pi/2, 0], [-1*math.pi/2, 0], [math.pi, 0], [0, math.pi/2], [0, -1*math.pi/2]]
        
        # self.p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    def load_urdfs(self, path):
        list_file = path  ##   
        if not os.path.exists(list_file):  
            raise shutil.Error  

        with open(list_file, 'r') as f:   
            while 1:  
                line = f.readline()   
                if not line:    
                    break   
                name, pos, ang, exp = line.split(', ')
                x, y, z = pos.split(' ')
                t1, t2, t3 = ang.split(' ')

                self.urdfs_list.append(models[name]) ##  获取模型列表，然后再获取几个参数
                self.urdfs_pos.append([float(x), float(y), float(z)])
                self.urdfs_ang.append([np.deg2rad(float(t1)), np.deg2rad(float(t2)), np.deg2rad(float(t3))])
                self.urdfs_exp.append(str(exp.strip('\n')))
    def _urdf_nums(self):
        return len(self.urdfs_list)
    
    def movecamera(self, x, y, z):
        """
        移动相机至指定位置
        x, y: 世界坐标系中的xy坐标
        """
        self.viewMatrix = self.p.computeViewMatrix([x, y, z], [x, y, 0], [0, 1, 0])   # 相机高度设置为0.7m

    def movecamera_position_rotation(self, x, y, z): 
        """ 
        移动相机至指定位置
        x, y: 世界坐标系中的xy坐标
        """
        # self.viewMatrix = self.p.computeViewMatrix([x, y, z], [x, y, 0], [0, 1, 0])   # 相机高度设置为0.7m
        self.viewMatrix = self.p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[x, y, 0],
        distance=0.4,  
        yaw = 0.25 * math.pi ,     ## 将物体绕Y轴旋转（localRotationY）
        pitch= math.pi,   ## 将物体绕X轴旋转（localRotationX）
        roll= -2 * math.pi,   ## 将物体绕Z轴旋转（localRotationZ
        upAxisIndex=2,
    )


    # 加载单物体
    def loadObjInURDF(self, urdf_file, idx,basePosition, render_n=0):  
        """
        以URDF的格式加载单个obj物体

        urdf_file: urdf文件
        idx: 物体id， 等于-1时，采用file
        render_n: 当前物体的渲染次数，根据此获取物体的朝向
        """
        # 获取物体文件
        if idx >= 0:
            self.urdfs_filename = [self.urdfs_list[idx]]
            self.objs_id = [idx]
        else:
            self.urdfs_filename = [urdf_file]
            self.objs_id = [-1]
        self.num_urdf = 1

        self.urdfs_id = []
        self.urdfs_xyz = []
        self.urdfs_ang = []
        self.urdfs_scale = []

        # 方向
        # baseEuler = [self.EulerRPList[render_n][0], self.EulerRPList[render_n][1], 0]
        # baseEuler = [self.EulerRPList[render_n][0], self.EulerRPList[render_n][1], random.uniform(0, 2*math.pi)]
        # baseEuler = [random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi)]
        # baseOrientation = self.p.getQuaternionFromEuler(baseEuler)
        # baseOrientation = [0, 0, 0, 1]    # 固定方向

        baseEuler = [0, 0, 0]  
        baseOrientation = self.p.getQuaternionFromEuler(baseEuler)

        # 随机位置
        # pos = 0.1 
        # basePosition = [random.uniform(-1 * pos, pos), random.uniform(-1 * pos, pos), random.uniform(0.1, 0.4)] 
        # basePosition = [random.uniform(-1 * pos, pos), random.uniform(-1 * pos, pos), -1*min_z] 
        # basePosition = [0.6 , 0 , 0.10] # 固定位置 (导入时药瓶的高度) 
        # basePosition = [0.6 , 0 , 0.1] # 固定位置 (导入时药瓶的高度) 
        basePosition = basePosition

        # 加载物体
        urdf_id = self.p.loadURDF(self.urdfs_filename[0], basePosition, baseOrientation)     # 4  5   6

        # 获取xyz和scale信息
        inf = self.p.getVisualShapeData(urdf_id)[0]

        self.urdfs_id.append(urdf_id)
        self.urdfs_xyz.append(inf[5]) 
        self.urdfs_scale.append(inf[3][0]) 

        ## 加载物体后等一会 ，物体可能会出现滑动等情况，等它稳定下来
        t = 0
        while True:
            p.stepSimulation()
            t += 1
            if t == 120:
                break
    
    # 加载多物体
    def loadObjsInURDF(self):   
        """ 
        以URDF的格式加载多个obj物体

        num: 加载物体的个数   5

        idx: 开始的id   0
            idx为负数时，随机加载num个物体
            idx为非负数时，从id开始加载num个物体
        """
        
        self.num_urdf = len(self.urdfs_list)

        # 获取物体文件
        
        self.urdfs_filename = self.urdfs_list
        self.objs_id = list(range(self.num_urdf))
        
        print('self.objs_id = \n', self.objs_id)

        self.urdfs_id = []
        self.urdfs_xyz = []
        self.urdfs_scale = []
        for i in range(self.num_urdf):
            # 随机位置
            pos = 0.1
            basePosition = self.urdfs_pos[i] # 位置

            baseEuler = self.urdfs_ang[i] # 方向
            baseOrientation = self.p.getQuaternionFromEuler(baseEuler)

            
            # 加载物体
            urdf_id = self.p.loadURDF(self.urdfs_filename[i], basePosition, baseOrientation)    

            # 使物体和机械手可以碰撞 
            # 机械手默认和所有物体不进行碰撞   
            ## 提前设置好机械手不与托盘之类的物体产生碰撞：通过在panda_gripper.urdf中第51行的collision group=“0” mask=“0” 来实现的。
            ## 通过程序再将机械手和物体之间添加上可碰撞：下方的代码。这样就会抓取物体并且不会碰到托盘。 
            if self.gripperId is not None:
                self.p.setCollisionFilterPair(urdf_id, self.gripperId, -1, 0, 1)
                self.p.setCollisionFilterPair(urdf_id, self.gripperId, -1, 1, 1)
                self.p.setCollisionFilterPair(urdf_id, self.gripperId, -1, 2, 1)

            # 获取xyz和scale信息，这里是获取物体信息以及一些后处理 
            inf = self.p.getVisualShapeData(urdf_id)[0]

            self.urdfs_id.append(urdf_id)
            self.urdfs_xyz.append(inf[5]) 
            self.urdfs_scale.append(inf[3][0]) 
            
            ## 加载物体后等一会 ，物体可能会出现滑动等情况，等它稳定下来
            t = 0
            while True:
                p.stepSimulation()
                t += 1
                if t == 120:
                    break


    def evalGrasp(self, z_thresh):
        """
        验证抓取是否成功
        如果某个物体的z坐标大于z_thresh，则认为抓取成功
        """
        for i in range(self.num_urdf):
            offset, _ =  self.p.getBasePositionAndOrientation(self.urdfs_id[i])
            if offset[2] >= z_thresh:
                return True
        print('!!!!!!!!!!!!!!!!!!!!! 失败 !!!!!!!!!!!!!!!!!!!!!')
        return False

    def evalGraspAndRemove(self, z_thresh, obj_id=None):
        """
        验证抓取是否成功，并删除抓取的物体
        如果某个物体的z坐标大于z_thresh，则认为抓取成功
        """
        if obj_id is None:
            for i in range(self.num_urdf):
                offset, _ =  self.p.getBasePositionAndOrientation(self.urdfs_id[i])
                if offset[2] >= z_thresh:
                    print('!!!!!!!!!!!!!!!!!!!!! SUCCESS !!!!!!!!!!!!!!!!!!!!!')
                    self.removeObjInURDF(i)
                    return True
        else:
            # offset, _ =  self.p.getBasePositionAndOrientation(self.urdfs_id[0])
            # if offset[2] >= z_thresh:
            #     self.removeObjInURDF(0)
            #     print('!!!!!!!!!!!!!!!!!!!!! SUCCESS !!!!!!!!!!!!!!!!!!!!!')
            #     return True
            # else:
            #     self.removeObjInURDF(0)
            for i in range(self.num_urdf):
                offset, _ =  self.p.getBasePositionAndOrientation(self.urdfs_id[i])
                
                if offset[2] >= z_thresh:
                    if i == 0: 
                        print('!!!!!!!!!!!!!!!!!!!!! SUCCESS !!!!!!!!!!!!!!!!!!!!!')
                        self.removeObjInURDF(0)
                        for i in range(self.num_urdf):
                            self.p.resetBasePositionAndOrientation(self.urdfs_id[i], posObj=self.urdfs_pos[i], ornObj=self.p.getQuaternionFromEuler(self.urdfs_ang[i])) 
                        return True
                    else:
                        for i in range(self.num_urdf):
                            self.p.resetBasePositionAndOrientation(self.urdfs_id[i], posObj=self.urdfs_pos[i], ornObj=self.p.getQuaternionFromEuler(self.urdfs_ang[i])) 
                        
            self.removeObjInURDF(0)   
            

                    

            
        print('!!!!!!!!!!!!!!!!!!!!! FAILURE !!!!!!!!!!!!!!!!!!!!!')
        
        return False
    

    def resetObjsPoseRandom(self):
        """
        随机重置物体的位置
        path: 存放物体位姿文件的文件夹
        """
        # 读取path下的objsPose.mat文件
        for i in range(self.num_urdf):
            pos = 0.1
            basePosition = [random.uniform(-1 * pos, pos), random.uniform(-1 * pos, pos), random.uniform(0.3, 0.6)]
            baseEuler = [random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi)]
            baseOrientation = self.p.getQuaternionFromEuler(baseEuler)
            self.p.resetBasePositionAndOrientation(self.urdfs_id[i], basePosition, baseOrientation)

            t = 0
            while True:
                p.stepSimulation()
                t += 1
                if t == 120:
                    break


    def removeObjsInURDF(self):
        """
        移除所有objs
        """
        for i in range(self.num_urdf):
            self.p.removeBody(self.urdfs_id[i])
        self.num_urdf = 0
        self.urdfs_id = []
        self.urdfs_xyz = []
        self.urdfs_pos = []
        self.urdfs_ang = []
        self.urdfs_scale = []
        self.urdfs_filename = []
        self.objs_id = []

    def removeObjInURDF(self, i):
        """
        移除指定的obj
        """
        self.num_urdf -= 1
        self.p.removeBody(self.urdfs_id[i])
        self.urdfs_id.pop(i)
        self.urdfs_xyz.pop(i)
        self.urdfs_pos.pop(i)
        self.urdfs_ang.pop(i)
        self.urdfs_scale.pop(i)
        self.urdfs_filename.pop(i)
        self.objs_id.pop(i)
    


    def renderCameraDepthImage(self):
        """
        渲染计算抓取配置所需的图像
        """
        # 渲染图像  它就会返回回来一个元组（图像宽度高度深度）
        img_camera = self.p.getCameraImage(IMAGEWIDTH, IMAGEHEIGHT, self.viewMatrix, self.projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL, shadow=0)
        w = img_camera[0]      # width of the image, in pixels
        h = img_camera[1]      # height of the image, in pixels
        dep = img_camera[3]    # depth data
        color = img_camera[2] 
        segm = img_camera[4] 

        color_image = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)  
        cv2.imwrite("./zxx_image.png",color_image)   ## ggcnn用的这里的图像。 

        # 获取深度图像 
        ## 不是单通道的深度图
        ## 所以需要进行一些转化，转化为可以使用的单通道的深度图（单位也要转化为m）
        depth = np.reshape(dep, (h, w))  # [40:440, 120:520]
        A = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * farPlane * nearPlane
        B = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * farPlane
        C = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * (farPlane - nearPlane)
        # im_depthCamera = A / (B - C * depth)  # 单位 m
        im_depthCamera = np.divide(A, (np.subtract(B, np.multiply(C, depth))))  # 单位 m
        return im_depthCamera

    def render_camera_D435(self, config):  ### 关于realsense相机的设置;    
        """Render RGB-D image with specified camera configuration."""

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1) 
        # lookdir = np.float32([0, 0, 1]).reshape(3, 1) 
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config["position"] + lookdir
        focal_len = config["intrinsics"][0, 0]
        znear, zfar = config["zrange"]
        viewm = p.computeViewMatrix(config["position"], lookat, updir) 
        fovh = (config["image_size"][0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = config["image_size"][1] / config["image_size"][0]
        projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = p.getCameraImage(
            width=config["image_size"][1],
            height=config["image_size"][0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=0,
            flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            # renderer = p.ER_BULLET_HARDWARE_OPENGL
            renderer = p.ER_TINY_RENDERER
        )
        
        # Get color image.
        color_image_size = (config["image_size"][0], config["image_size"][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config["noise"]:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (config["image_size"][0], config["image_size"][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear)
        depth = (2.0 * znear * zfar) / depth
        if config["noise"]:
            depth += self._random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm




    def renderCameraMask(self):
        """
        渲染计算抓取配置所需的图像 
        """
        # 渲染图像
        img_camera = self.p.getCameraImage(IMAGEWIDTH, IMAGEHEIGHT, self.viewMatrix, self.projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL, shadow=0)
        w = img_camera[0]      # width of the image, in pixels
        h = img_camera[1]      # height of the image, in pixels
        # rgba = img_camera[2]    # color data RGB
        # dep = img_camera[3]    # depth data
        mask = img_camera[4]    # mask data

        # 获取分割图像
        im_mask = np.reshape(mask, (h, w)).astype(np.uint8)
        im_mask[im_mask > 2] = 255
        return im_mask


    def gaussian_noise(self, im_depth):
        """
        在image上添加高斯噪声，参考dex-net代码

        im_depth: 浮点型深度图，单位为米
        """
        gamma_shape = 1000.00
        gamma_scale = 1 / gamma_shape
        gaussian_process_sigma = 0.002  # 0.002
        gaussian_process_scaling_factor = 8.0   # 8.0

        im_height, im_width = im_depth.shape
        
        # 1
        # mult_samples = ss.gamma.rvs(gamma_shape, scale=gamma_scale, size=1) # 生成一个接近1的随机数，shape=(1,)
        # mult_samples = mult_samples[:, np.newaxis]
        # im_depth = im_depth * np.tile(mult_samples, [im_height, im_width])  # 把mult_samples复制扩展为和camera_depth同尺寸，然后相乘
        
        # 2
        gp_rescale_factor = gaussian_process_scaling_factor     # 4.0
        gp_sample_height = int(im_height / gp_rescale_factor)   # im_height / 4.0
        gp_sample_width = int(im_width / gp_rescale_factor)     # im_width / 4.0
        gp_num_pix = gp_sample_height * gp_sample_width     # im_height * im_width / 16.0
        gp_sigma = gaussian_process_sigma
        gp_noise = ss.norm.rvs(scale=gp_sigma, size=gp_num_pix).reshape(gp_sample_height, gp_sample_width)  # 生成(均值为0，方差为scale)的gp_num_pix个数，并reshape
        # print('高斯噪声最大误差:', gp_noise.max())
        gp_noise = imresize(gp_noise, gp_rescale_factor, interp="bicubic")  # resize成图像尺寸，bicubic为双三次插值算法
        # gp_noise[gp_noise < 0] = 0
        # camera_depth[camera_depth > 0] += gp_noise[camera_depth > 0]
        im_depth += gp_noise

        return im_depth

    def add_noise(self, img):
        """
        添加高斯噪声和缺失值
        """
        img = self.gaussian_noise(img)    # 添加高斯噪声
        # 补全
        # img = inpaint(img, missing_value=0)
        return img