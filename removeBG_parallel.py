from rembg import remove, new_session
import sys, os, shutil, cv2, re
import numpy as np
import multiprocessing
import tqdm
import base64

n=0

#Create mask as <filename>_mask.png at pathToFolder
def createMask(dirpath, mask_abs_path):
    extensions = ['.jpg','.JPG','.png','.PNG']
    paths = [str for str in os.listdir(dirpath) if any(sub in str for sub in extensions)]
    no_of_image = len(paths)
    paths = [(dirpath, str, mask_abs_path, no_of_image) for str in sorted(paths) if 'mask' not in str]
    pool = multiprocessing.Pool()
    pool.starmap(run_mask, paths)
    pool.close()
    pool.join
    #for path in paths:
    #    run_mask(*path)

def run_mask(dirpath, file, mask_abs_path, no_of_image):
    input_path = os.path.join(dirpath,file)
    #ext = file.split('.')[-1]
    ext = 'png'
    output_path = os.path.join(dirpath, file.split('.')[0]+f'_mask.{ext}')
    finalmask_path = os.path.join(mask_abs_path, file.split('.')[0]+f'_mask.{ext}')
    maskprocess=False
    if not os.path.exists(finalmask_path):
        maskprocess=True
        with open(input_path, 'rb') as i:
            input = i.read()
        session=new_session('u2netp')
        output_byte = remove(data=input, session=session, only_mask=True)
        convertMask(output_path,output_byte)  
    
    global n
    if maskprocess:
        print(file,'masked')
        #print(file,'masked','('+str(n+1)+'/'+str(no_of_image)+')')
    else:
        print(os.path.basename(finalmask_path),'already exists')
        #print(finalmask_path,'already exists','('+str(n+1)+'/'+str(no_of_image)+')')
    n+=1
    
    return 0
    
#convert and replaces mask to single channel, black n white
def convertMask(image_path, image_byte):
    #image_path = path
    im = np.frombuffer(image_byte, np.uint8)
    im = cv2.imdecode(im, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = np.greater_equal(im, 40) #10
    im = np.where(im, 255, 0)
    cv2.imwrite(image_path, im)

#move masks to pathToMask directory
def moveMask(path0, path1):
    if not os.path.exists(path1):
        os.mkdir(path1)
    for file in os.listdir(path0):
        if '_mask' in file:
            input_path = os.path.join(path0, file)
            output_path = os.path.join(path1,file)
            if not os.path.exists(output_path):
                shutil.move(input_path,output_path)

def main():
    pathToFolder = sys.argv[1]
    if not os.path.exists(pathToFolder):
        print('rembg: folder does not exists: ',pathToFolder)
        sys.exit(1)
    if len(sys.argv) > 2:
        pathToMask = sys.argv[2]
    else:
        parent_Folder = os.path.join(pathToFolder, os.pardir)
        pathToMask = os.path.join(parent_Folder,'masks')
    print('rembg: image directory:', pathToFolder)
    print('rembg: mask directory:',pathToMask)    
    createMask(pathToFolder,pathToMask)
    moveMask(pathToFolder, pathToMask)
    print('rembg: completed')

main()