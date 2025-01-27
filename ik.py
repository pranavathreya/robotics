import numpy as np

def get_ik(end_effector_pos): 
    x = end_effector_pos[0] 
    y = end_effector_pos[1]
    z = end_effector_pos[2] 
    alpha = end_effector_pos[3]
    #links are in mm 
    L1 = 77   
    L2 = 130  
    L3 = 124  
    L4 = 126  
    l21 = 128 
    l22 = 24  

    joint_angles = []

    r = np.sqrt(x**2 + y**2)
    rw = r - L4 * np.cos(np.radians(alpha))
    zw = z - L1 - L4 * np.sin(np.radians(alpha))
    dw = np.sqrt(rw**2 + zw**2)
    mu = np.arctan2(zw, rw)
    
    cos_beta = (L2**2 + L3**2 - dw**2) / (2 * L2 * L3)
    sin_beta = np.sqrt(1 - cos_beta**2)
    beta = [np.atan2(sin_beta, cos_beta), np.atan2(-sin_beta, cos_beta)]

    cos_gamma = (dw**2 + L2**2 - L3**2) / (2 * dw * L2)
    sin_gamma = np.sqrt(1 - cos_gamma**2)
    gamma = [np.atan2(sin_gamma, cos_gamma), np.atan2(-sin_gamma, cos_gamma)]

    delta = np.arctan2(l22, l21)
    
    for i in range(2):
        theta_1 = np.arctan2(y, x)
        theta_2 = (np.pi / 2) - delta - gamma[i] - mu
        theta_3 = (np.pi / 2) + delta - beta[i]
        theta_4 = -np.radians(alpha) - theta_2 - theta_3

        joint_angles.append(
            np.degrees([theta_1, theta_2, theta_3, theta_4]))

    best_pos_index = 0; 

    solution = joint_angles[0]

    if np.any(np.isnan(solution)):
        print("Solution is invalid:", solution)
        raise ValueError
    if np.any(np.logical_or(solution > 180, solution < -180)):
        print("Solution is unreachable:", solution)

    print(solution)
    return solution

