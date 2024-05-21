import random
import torch
import numpy as np

def change_blur_magnitude_mean10(blur_condition):
    device = blur_condition.device
    magnitude = blur_condition[:, 2]
    blur_mean_magnitude = torch.mean(magnitude)
    # ------- change magnitude--------------  
    if blur_mean_magnitude < 0.1:
        if random.randint(0, 1) == 1:
            #Transform blur into clarity, and clarity into blur.
            value = random.uniform(0.1, 0.2)
            magnitude = value - magnitude
            magnitude[magnitude < 0] = 0
        else:
            #Increase the degree of global blur.  
            value = random.uniform(0.03, 0.12)
            magnitude = torch.full_like(magnitude, value)
    elif random.randint(0, 3) == 0:
        # Increase the degree of global blur. 
        value = random.uniform(0.03, 0.12)
        magnitude = torch.full_like(magnitude, value)
    elif blur_mean_magnitude > 0.3:
        #Reduce the degree of blur.
        value = random.uniform(0.1, 0.3)
        magnitude = magnitude * value
    else:
        #Randomly adjust the degree of blur, increasing or decreasing it
        value = random.uniform(0.2, 0.4)
        magnitude = magnitude * value
        magnitude[magnitude > 1] = 1
    
    blur_condition[:, 2] = magnitude
    
    return blur_condition.to(device)

def change_blur_magnitude_mean20(blur_condition):
    device = blur_condition.device
    magnitude = blur_condition[:, 2]
    blur_mean_magnitude = torch.mean(magnitude)
    # ------- change magnitude--------------  
    if blur_mean_magnitude < 0.075:
        if random.randint(0, 1) == 1:
            #Transform blur into clarity, and clarity into blur.
            value = random.uniform(0.15, 0.2)
            magnitude = value - magnitude
            magnitude[magnitude < 0] = 0
        else:
            # Increase the degree of global blur. 
            value = random.uniform(0.075, 0.15)
            magnitude = torch.full_like(magnitude, value)
    elif random.randint(0, 3) == 0:
        # Increase the degree of global blur. 
        value = random.uniform(0.05, 0.2)
        magnitude = torch.full_like(magnitude, value)
    elif blur_mean_magnitude > 0.3:
        #Reduce the degree of blur.
        value = random.uniform(0.1, 0.9)
        magnitude = magnitude * value
    else:
        #Randomly adjust the degree of blur, increasing or decreasing it
        value = random.uniform(0.5, 2)
        magnitude = magnitude * value
        magnitude[magnitude > 1] = 1
    
    blur_condition[:, 2] = magnitude
    
    return blur_condition.to(device)

def change_blur_magnitude_mean40(blur_condition):
    device = blur_condition.device
    magnitude = blur_condition[:, 2]
    blur_mean_magnitude = torch.mean(magnitude)

    if blur_mean_magnitude < 0.1:
        if random.randint(0, 1) == 1:
            #Transform blur into clarity, and clarity into blur.
            value = random.uniform(0.2, 0.3)
            magnitude = value - magnitude
            magnitude[magnitude < 0] = 0
        else:
            # Increase the degree of global blur.
            value = random.uniform(0.2, 0.3)
            magnitude = torch.full_like(magnitude, value)
    elif random.randint(0, 3) == 0:
        # Increase the degree of global blur. 
        value = random.uniform(0.2, 0.4)
        magnitude = torch.full_like(magnitude, value)
    elif blur_mean_magnitude > 0.4:
        #Reduce the degree of blur.
        value = random.uniform(0.8, 1.2)
        magnitude = magnitude * value
        magnitude[magnitude > 1] = 1
    elif blur_mean_magnitude > 0.2:
        #Reduce the degree of blur.
        value = random.uniform(1.2, 1.5)
        magnitude = magnitude * value
        magnitude[magnitude > 1] = 1
    else:
        #Randomly adjust the degree of blur, increasing or decreasing it
        value = random.uniform(1.5, 2.5)
        magnitude = magnitude * value
        magnitude[magnitude > 1] = 1
    
    blur_condition[:, 2] = magnitude

    return blur_condition.to(device)

def change_blur_magnitude_mean60(blur_condition):
    device = blur_condition.device
    magnitude = blur_condition[:, 2]
    blur_mean_magnitude = torch.mean(magnitude)

    if blur_mean_magnitude < 0.1:
        if random.randint(0, 1) == 1:
            #Transform blur into clarity, and clarity into blur.
            value = random.uniform(0.35, 0.45)
            magnitude = value - magnitude
            magnitude[magnitude < 0] = 0
        else:
            # Increase the degree of global blur.
            value = random.uniform(0.3, 0.5)
            magnitude = torch.full_like(magnitude, value)
    elif random.randint(0, 3) == 0:
        # Increase the degree of global blur. 
        value = random.uniform(0.3, 0.5)
        magnitude = torch.full_like(magnitude, value)
    elif blur_mean_magnitude > 0.4:
        #Reduce the degree of blur.
        value = random.uniform(0.8, 1.2)
        magnitude = magnitude * value
        magnitude[magnitude > 1] = 1
    elif blur_mean_magnitude > 0.2:
        #Reduce the degree of blur.
        value = random.uniform(1.5, 2)
        magnitude = magnitude * value
        magnitude[magnitude > 1] = 1
    else:
        #Randomly adjust the degree of blur, increasing or decreasing it
        value = random.uniform(2.5, 4.5)
        magnitude = magnitude * value
        magnitude[magnitude > 1] = 1
    
    blur_condition[:, 2] = magnitude

    return blur_condition.to(device)

def change_blur_magnitude_mean80(blur_condition):
    device = blur_condition.device
    magnitude = blur_condition[:, 2]
    blur_mean_magnitude = torch.mean(magnitude)

    if blur_mean_magnitude < 0.1:
        if random.randint(0, 1) == 1:
            #Transform blur into clarity, and clarity into blur.
            value = random.uniform(0.5, 0.7)
            magnitude = value - magnitude
            magnitude[magnitude < 0] = 0
        else:
            # Increase the degree of global blur.
            value = random.uniform(0.5, 0.7)
            magnitude = torch.full_like(magnitude, value)
    elif random.randint(0, 3) == 0:
        # Increase the degree of global blur. 
        value = random.uniform(0.5, 0.7)
        magnitude = torch.full_like(magnitude, value)
    elif blur_mean_magnitude > 0.4:
        #Reduce the degree of blur.
        value = random.uniform(1.2, 1.5)
        magnitude = magnitude * value
        magnitude[magnitude > 1] = 1
    elif blur_mean_magnitude > 0.2:
        #Reduce the degree of blur.
        value = random.uniform(2, 3)
        magnitude = magnitude * value
        magnitude[magnitude > 1] = 1
    else:
        #Randomly adjust the degree of blur, increasing or decreasing it
        value = random.uniform(4, 6)
        magnitude = magnitude * value
        magnitude[magnitude > 1] = 1
    
    blur_condition[:, 2] = magnitude

    return blur_condition.to(device)

def change_orientation(blur_condition):
    # ------- change orientation--------------
    if random.randint(0, 1) == 0:
            if random.randint(0, 1) == 1:
                #Alter in the x-direction.
                blur_condition[:, 0] = -blur_condition[:, 0]
            if random.randint(0, 1) == 1:
                #Alter in the y-direction.
                blur_condition[:, 1] = -blur_condition[:, 1]
    else:
        #Fixed in four directions: up, down, left, and right.
        orientations = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        sample = random.choice(orientations)
        blur_condition[:, 0] = torch.full_like(blur_condition[:, 0], sample[0])
        blur_condition[:, 1] = torch.full_like(blur_condition[:, 1], sample[1])
    
    return blur_condition

def rotation_matrix(angle):
    rad = np.radians(angle)
    
    cos_theta = np.cos(rad)
    sin_theta = np.sin(rad)
    rot_matrix = np.array([[cos_theta, -sin_theta],
                           [sin_theta, cos_theta]])
    
    return rot_matrix

def change_fixed_orientation(blur_condition, choice):
    device = blur_condition.device
    # ------- change orientation--------------
    if choice == 0 or choice == 1:
        vectors = blur_condition[:, :2].clone().cpu().numpy()

        vectors = vectors.transpose((0, 2, 3, 1)) 
        vectors_origin_shape = vectors.shape
        
        vectors = vectors.reshape((-1, 2))
        rot_matrix = rotation_matrix(120 * (choice + 1))
        
        # use rotate matrix
        rotated_vectors = (rot_matrix@vectors.T).T
        rotated_vectors = torch.from_numpy(rotated_vectors.reshape(vectors_origin_shape).transpose((0, 3, 1, 2))).to(device)
        
        blur_condition[:, :2] = rotated_vectors
    else:
        if choice == 2:
            sample = [1, 0]
        elif choice == 3:
            sample = [-1, 0]
        else:
            orientations = [[0, 1], [0, -1]]
            sample = random.choice(orientations)
        blur_condition[:, 0] = torch.full_like(blur_condition[:, 0], sample[0])
        blur_condition[:, 1] = torch.full_like(blur_condition[:, 1], sample[1])
    
    return blur_condition

def change_degree_orientation(blur_condition, rotate_degree):
    device = blur_condition.device
    # -------  change orientation--------------
    vectors = blur_condition[:, :2].clone().cpu().numpy()

    vectors = vectors.transpose((0, 2, 3, 1)) 
    vectors_origin_shape = vectors.shape
    
    vectors = vectors.reshape((-1, 2))
    rot_matrix = rotation_matrix(rotate_degree)
    
    # use rotate matrix
    rotated_vectors = (rot_matrix@vectors.T).T
    rotated_vectors = torch.from_numpy(rotated_vectors.reshape(vectors_origin_shape).transpose((0, 3, 1, 2))).to(device)
    
    blur_condition[:, :2] = rotated_vectors
    
    return blur_condition

def select_condition_strategy(blur_condition, strategy, choice_num=None, change_base=0):
    if len(strategy) == 0:
        return blur_condition
    
    # ------- change magnitude--------------  
    M_choice = -1
    if 'ALLM' in strategy:
        if choice_num is None:
            M_choice = random.randint(0, 4)
        else:
            M_choice = choice_num % 5

    new_condition = blur_condition
    if 'M20' in strategy or M_choice == 0: 
        new_condition = change_blur_magnitude_mean20(new_condition)
    elif 'M40' in strategy or M_choice == 1:
        new_condition = change_blur_magnitude_mean40(new_condition)
    elif 'M60' in strategy or M_choice == 2:
        new_condition = change_blur_magnitude_mean60(new_condition)
    elif 'M80' in strategy or M_choice == 3:
        new_condition = change_blur_magnitude_mean80(new_condition)
    elif 'M30' in strategy or M_choice == 4:
        if random.randint(0, 1) == 1:
            new_condition = change_blur_magnitude_mean20(new_condition)
        else:
            new_condition = change_blur_magnitude_mean40(new_condition)
    elif 'M10' in strategy:
        new_condition = change_blur_magnitude_mean10(new_condition)

    # ------- change orientation--------------
    if 'ALLO' in strategy:
        if choice_num is None:
            O_choice = random.randint(0, 4)
        else:
            O_choice = (choice_num + change_base) % 5
        new_condition = change_fixed_orientation(new_condition, O_choice)
    elif 'O' in strategy:
        new_condition = change_orientation(new_condition)
    elif 'RO':
        random_rotate_degree = random.randint(1, 359)
        new_condition = change_degree_orientation(new_condition, random_rotate_degree)
    elif '60O':
        O_choice = (choice_num + change_base) % 5 + 1
        rotate_degree = O_choice * 60
        new_condition = change_degree_orientation(new_condition, rotate_degree)
    elif '30O':
        rotate_degree = random.randint(1, 12) * 30
        new_condition = change_degree_orientation(new_condition, rotate_degree)
    return new_condition