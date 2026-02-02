import cv2
import os

def split_image_width_cv(image_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    left_half = img[:, :w//2]
    right_half = img[:, w//2:]

    return (left_half, right_half)

    
unprocessed_data = os.listdir('dataset_unprocessed')

unprocessed_data_folder_path = 'dataset_unprocessed'

for i in unprocessed_data:
   img_path =  os.path.join(unprocessed_data_folder_path,i)
   
   no_pain,pain = split_image_width_cv(img_path)
   no_pain_img_path = os.path.join('dataset_processed','no_pain',i)
   pain_img_path = os.path.join('dataset_processed','pain',i)
   print(pain_img_path,no_pain_img_path)

   cv2.imwrite(no_pain_img_path, no_pain)
   cv2.imwrite(pain_img_path, pain)




