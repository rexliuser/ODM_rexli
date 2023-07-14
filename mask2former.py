from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch
from PIL import Image
import numpy as np
import os, shutil
import argparse
from tqdm import tqdm
import csv

parser = argparse.ArgumentParser(
        prog='mask2former.py',
        description='Run masks using mask2former. Input a folder.'
    )
parser.add_argument('image_folder_path')
args = parser.parse_args()
image_folder_path=os.path.abspath(args.image_folder_path)
extensions = ['jpg','JPG','png','PNG']
object_filter = ['rug','wall','floor']

device = "cuda" if torch.cuda.is_available() else "cpu"
print('using',device)

def create_masks(create_dir_path,image_path,processor,model):
    image = Image.open(image_path)
    folder_path = create_dir_path
    
    x,y=image.size
    x_array = np.tile(np.arange(x),(y,1))
    y_array = np.tile(np.arange(y),(x,1)).T

    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    mask_details = []
    ##############################################
    for i,seg in enumerate(prediction['segments_info']):
        label_id = seg['label_id']
        name = model.config.id2label[label_id]
        name = name.replace('merged','').replace('-','')
        if not any([f in name for f in object_filter]):
        
            tensor = prediction['segmentation'].to('cpu').numpy()
            tensor = np.where(tensor==i+1,1,0)
            
            n = np.count_nonzero(tensor)
            percent = n/tensor.size
                       
            if percent > 0.03 and percent<0.8:
                
                ax = tensor * x_array
                ay = tensor * y_array
                #calculate average distance
                mx = ax.flatten()
                my = ay.flatten()
                avg_x = np.mean(mx[mx!=0])/x
                avg_y = np.mean(my[my!=0])/y
                dist = np.linalg.norm(np.array([avg_x,avg_y/1.5])-np.array([0.5,0.5])) #x axis has more influence than y axis
                
                #calculate dispersion
                mx = np.where(ax==0,np.nan,ax)/x-avg_x
                my = np.where(ay==0,np.nan,ay)/y-avg_y
                dispersion = np.nanmean(np.sqrt(np.square(mx)+np.square(my)))
                #print('dispersion:',dispersion)
                
                

                #print(name+str(i),str(percent)+'%')
                tensor = (tensor*255).astype(np.uint8)
                im = Image.fromarray(tensor, mode='L')
                im.save(f'{folder_path}/mask{i}.png')
                #print('dist_score:',dist)
                #print('----')
                
                if 'table' in name:
                    object_offset = 0.7
                else:
                    object_offset = 1
                
                dict = {"name":f"mask{i}.png",
                        "object":name,
                        "dist_score":dist,
                        'dispersion':dispersion,
                        'object_offset':object_offset,
                        "coverage":percent}
                mask_details.append(dict)
            
    output_dict = {'image_path':image_path,
                   'masks':mask_details}
    return output_dict

def clear_subfolders(image_folder_path,mode=1):
    # mode1: folders+masks    mode2:folders only
    print(f"Removing subfolders in {image_folder_path}")
    for file in os.listdir(image_folder_path):
        if os.path.basename(image_folder_path)=='images':
            if os.path.isdir(os.path.join(image_folder_path,file)):
                print(f'-- removing mask folder:{file}')
                shutil.rmtree(os.path.join(image_folder_path,file))
            elif '_mask' in file and mode==1:
                print(f'-- removing mask:{file}')
                os.remove(os.path.join(image_folder_path,file))

def create_masks():
    #create mask folder
    mask_folder_path = os.path.join(image_folder_path,'masks')
    os.mkdir(mask_folder_path)
    
    print('Masking Images...')
    for img in tqdm([file for file in sorted(os.listdir(image_folder_path))[:] if file[-3:] in extensions]):
        image_path = os.path.join(image_folder_path,img)
        image_name = img.split('.')[0]
        submask_folder_path = os.path.join(mask_folder_path,image_name)
        os.mkdir(submask_folder_path)
        
        image = Image.open(image_path)
        
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]], label_ids_to_fuse=[None])[0]
        tensor = prediction['segmentation'].to('cpu').numpy()
        
        dict_list = []
        
        for i,seg in enumerate(prediction['segments_info']):
            label_id = seg['label_id']
            object_type = model.config.id2label[label_id]
            object_type = object_type.replace('merged','').replace('-','').replace('other','')
            
            im_array = np.where(tensor==i+1,1,0)
            percent = np.count_nonzero(im_array)/im_array.size
            if percent > 0.01 and not any([banned_object in object_type for banned_object in object_filter]):
                dict_list.append(get_dict(im_array,i,object_type))
                mask_name = str(i)
                im_array = np.where(im_array>0,255,0).astype(np.uint8)
                Image.fromarray(im_array, mode='L').save(f'{submask_folder_path}/{mask_name}.png')
            
                #create an inverse mask
                #if object_type not in object_filter:
                #    im_array = np.where(tensor==i+1,0,1)
                #    percent = np.count_nonzero(im_array)/im_array.size
                #    if percent < 0.7:
                #        dict_list.append(get_dict(im_array,i+100))
                #        mask_name = str(i+100)
                #        im_array = np.where(im_array>0,255,0).astype(np.uint8)
                #        Image.fromarray(im_array, mode='L').save(f'{submask_folder_path}/{mask_name}.png')
                
        if not dict_list:
            dict_list.append(get_dict(np.zeros(im_array.shape),0,'null'))
            Image.fromarray(np.zeros(im_array.shape), mode='L').save(f'{submask_folder_path}/0.png')
            
        csv_path = os.path.join(submask_folder_path,'metadata.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=dict_list[0].keys())
            writer.writeheader()
            writer.writerows(dict_list)

def get_dict(im_array,i,object_type):
    x=np.argmax((im_array==1).any(axis=0))
    y=np.argmax((im_array==1).any(axis=1))
    w=im_array.shape[1]-np.argmax((np.fliplr(im_array==1)).any(axis=0))-x
    h=im_array.shape[0]-np.argmax((np.flipud(im_array==1)).any(axis=1))-y
    dict = {
        'id':i,
        'area':np.count_nonzero(im_array),
        'bbox_x0':x,
        'bbox_y0':y,
        'bbox_w':w,
        'bbox_h':h,
        'point_input_x':0,
        'point_input_y':0,
        'predicted_iou':0,
        'stability_score':0,
        'crop_box_x0':0,
        'crop_box_y0':0,
        'crop_box_w':im_array.shape[1],
        'crop_box_h':im_array.shape[0],
        'object_type':object_type
    }
    return dict

def main(image_folder_path):
    
    print('Clearning Folders...')
    clear_subfolders(image_folder_path)
    
    create_masks()
    


print('Importing AutoImageProcessor and Mask2FormerForUniversalSegmentation...')
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic").to(device)

main(image_folder_path)