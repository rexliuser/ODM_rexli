#downscale image to mp
#input: dir or img
import argparse, os, math, shutil
from PIL import Image

parser = argparse.ArgumentParser(
    prog = 'downscale_img.py',
    description = 'downscale image to mp, input dir or image. Default 1MP'
                                 )
parser.add_argument('input_path')
parser.add_argument('-r','--resolution',dest='resolution',help='target resolution of image. Default 1MP', default=1.0)
parser.add_argument('-o','--output-dir',dest='output_dir',help='Output directory or image name. Default input_dir', default='')
args = parser.parse_args()

input_path = args.input_path
output_dir = args.output_dir if args.output_dir!='' else input_path
res=float(args.resolution)
extensions = ['.jpg','.JPG','png','PNG']

if os.path.isdir(input_path):
    imgs = [img for img in os.listdir(input_path) if os.path.splitext(img)[1] in extensions]
    input_files = [os.path.join(input_path,img) for img in imgs]
    output_files = [os.path.join(output_dir,f"{img.split('.')[0]}_dsimg.{img.split('.')[1]}") for img in imgs]
    
    for f in [os.path.join(output_dir,file) for file in os.listdir(output_dir) if '_dsimg' in file]:
        os.remove(f)
        print('Removed ',f)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
else:
    input_files = [input_path]
    output_files = [output_dir]
    
#loop through input_files
    #read, compress, save image
for i, img_path in enumerate(input_files):
    n=len(input_files)
    im = Image.open(img_path)
    if 'exif' in im.info.keys():
        metadata = im.info['exif']
    else:
        metadata = b''
    size=im.size
    
    if int(res)==0 or res>size[0]*size[1]:
        target_size = size
    else:
        div = math.sqrt(im.size[0]*im.size[1]/(res*1e06))
        target_size = (int(im.size[0]/div),int(im.size[1]/div))
    
    im = im.resize(target_size)
    im.save(output_files[i], exif=metadata)
    print('Downscale',os.path.basename(img_path),'from',size,'to',target_size,f'({i+1}/{n})')

print('Downscale_images Completed.')