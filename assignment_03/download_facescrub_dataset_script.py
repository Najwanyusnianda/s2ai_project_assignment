print("=========================run==============")

import os
import random
import shutil
import numpy as np
import pandas as pd
import requests
import cv2
from mtcnn import MTCNN
from imutils import paths
from tqdm import tqdm
from pathlib import Path
from deepface import DeepFace
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)



# Import metadata
facescrub_df_actor = pd.read_csv('faceScrub/facescrub_actors.txt', delimiter='\t', header=None)
facescrub_df_actress = pd.read_csv('faceScrub/facescrub_actresses.txt', delimiter='\t', header=None)

# Combine dataframe
facescrub_df = pd.concat([facescrub_df_actor, facescrub_df_actress], axis=0)
print(f"Number of rows (images): {len(facescrub_df)}")

# Example function to get train images
def get_images(folder_path):
    image_paths = list(paths.list_images(folder_path))
    return [{"person": os.path.basename(os.path.dirname(p)), "path": p} for p in image_paths]

train_images_set = get_images('dataset/stagging/sampleset')
train_dict = {p["person"]: p["path"] for p in train_images_set}
trp = list(train_dict.keys())
print(trp)
print(f"len trp={len(trp)}")
need_p = 200 - len(trp)

# Function to download and detect faces
def download_and_detect_faces(url, filename):
    detector = MTCNN()
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Convert to numpy array 
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
        # Check if image is valid
        if image is None:
            return 0

        # Detect faces in the image
        result = detector.detect_faces(image)
        if not len(result) == 1:
            return 0
            
        # Save the image if faces detected
        cv2.imwrite(filename, image)
        return 1
        
    except (requests.exceptions.RequestException, cv2.error):
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except OSError:
                pass
        return 0
    except Exception:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except OSError:
                pass
        return 0

# Function to get image samples
def get_image_sample(num_person, ntrain_person, ntest_person, nval_person, facescrub_df):
    current_num_person = 0
    list_person = trp

    while current_num_person < num_person:
        current_ntrain_person = 0
        current_ntest_person = 0
        current_nval_person = 0

        random_num = np.random.randint(0, facescrub_df.shape[0])
        current_person = facescrub_df.iloc[random_num, 0]
        
        if current_person in list_person:
            continue

        df_persons = facescrub_df[facescrub_df[0] == current_person]
        
        train_path = f'dataset/stagging/sampleset/{current_person}'
        val_path = f'dataset/stagging/validationset/{current_person}'
        test_path = f'dataset/stagging/testingset/{current_person}'

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        print(f"Downloading images for person={current_person}, num_images={df_persons.shape[0]}")
            
        list_index_person = []
        while current_ntrain_person < ntrain_person:
            i = np.random.randint(0, df_persons.shape[0])
            if i in list_index_person:
                continue
            list_index_person.append(i)
            url = df_persons.iloc[i, 3]
            isdownload = download_and_detect_faces(url, f"{train_path}/{current_person}_{current_ntrain_person+1}.jpg")
            if isdownload == 1:
                current_ntrain_person += 1

        list_person.append(current_person)
        current_num_person += 1
        print(f"=== [{current_num_person}/{num_person}] Person added: {current_person} ===")

# Run the function
get_image_sample(num_person=need_p, ntrain_person=20, ntest_person=5, nval_person=5, facescrub_df=facescrub_df)
