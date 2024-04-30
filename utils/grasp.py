import numpy as np

def asGraspRectangle(grasp):
    x1_, y1_, angle, x2_, y2_ = grasp
    cx, cy = round((x1_ + x2_)/2), round((y1_ + y2_)/2)
    w, h = np.abs(x2_ - x1_), np.abs(y2_ - y1_)

    ang = np.deg2rad(angle)

    x1 = round(cx + (w/2)*np.cos(ang) - (h/2)*np.sin(ang))
    y1 = round(cy + (w/2)*np.sin(ang) + (h/2)*np.cos(ang))

    x2 = round(cx + (w/2)*np.cos(ang) + (h/2)*np.sin(ang))
    y2 = round(cy + (w/2)*np.sin(ang) - (h/2)*np.cos(ang))

    x3 = round(cx - (w/2)*np.cos(ang) + (h/2)*np.sin(ang))
    y3 = round(cy - (w/2)*np.sin(ang) - (h/2)*np.cos(ang))

    x4 = round(cx - (w/2)*np.cos(ang) - (h/2)*np.sin(ang))
    y4 = round(cy - (w/2)*np.sin(ang) + (h/2)*np.cos(ang))

    return np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]).astype(np.float32)

def asGrasp(ref_graspRectangle, im_size=(480, 640)): # p1, p2, p3, p4 -> xmin, ymin, theta, xmax, ymax
    ref_graspRectangle = np.swapaxes(ref_graspRectangle, 1, 2) #[[x1, y1], [x2, y2], [x3, y3], [x4, y4]] -> [[x1, x2, x3, x4], [y1, y2, y3, y4]]
    A = ref_graspRectangle 
    xy_ctr = np.sum(A, axis=2) / 4
    x_ctr = xy_ctr[:, 0]
    y_ctr = xy_ctr[:, 1]

    height = np.sqrt(np.sum((A[:, :, 0] - A[:, :, 1]) ** 2, axis=1))
    width = np.sqrt(np.sum((A[:, :, 1] - A[:, :, 2]) ** 2, axis=1))
    theta = np.zeros((A.shape[0]), dtype=np.int32)

    # theta = np.arctan((A[:, 1, 1] - A[:, 1, 0]) / (A[:, 0, 1] - A[:, 0, 0] + 1e-12)) # y2 - y1 / x2 - x1
    dx = A[:, 0, 1] - A[:, 0, 2] # x2 - x3
    dy = A[:, 1, 1] - A[:, 1, 2] # y2 - y3
    theta = (np.arctan2(dy, dx) + np.pi/2) % np.pi - np.pi/2
    

    x_min = x_ctr - width / 2
    x_max = x_ctr + width / 2
    y_min = y_ctr - height / 2
    y_max = y_ctr + height / 2

    x_coords = np.vstack((x_min, x_max))
    y_coords = np.vstack((y_min, y_max))

    mat = np.asarray((np.all(x_coords > im_size[1], axis=0), np.all(x_coords < 0, axis=0),
                        np.all(y_coords > im_size[0], axis=0), np.all(y_coords < 0, axis=0)))

    fail = np.any(mat, axis=0)
    correct_idx = np.where(fail == False)
    
    theta_deg = np.rad2deg(theta)
    # print(theta_deg)
    ret_value = (
    x_min[correct_idx], y_min[correct_idx], theta_deg[correct_idx], x_max[correct_idx], y_max[correct_idx])
    ret_value = list(zip(*ret_value))
    return ret_value 