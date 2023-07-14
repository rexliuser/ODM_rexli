import os, sys, random
import imageio, math
from PIL import Image
import argparse
import multiprocessing
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Extract frames from video(s) and downscale output frames (optional)')
parser.add_argument('input_path',type=str,help='Input directory with video(s) (required)')
parser.add_argument('--o',dest='o',type=str,help='Output directory')
parser.add_argument('--f',dest='f',type=int,help='target no. of frames to extract (multi-clip inclusive)',default=150)
parser.add_argument('--r',dest='r',type=float,help='target resolution in MP (float)',default=0)
parser.add_argument('--p',dest='p',type=float,help='propability of randomly further downscale images by 2',default=0)

args = parser.parse_args()
input_path = args.input_path
output_path = args.o if args.o is not None else input_path
target_frames = args.f
target_res = args.r * 1e06
res_ratio = args.p


formats = ['mp4','MP4','mov','MOV']
if not os.path.exists(output_path):
    os.mkdir(output_path)
else:
    for item in os.listdir(output_path):
        if 'frame' in item:
            path = os.path.join(output_path,item)
            os.remove(path)
            print('removed',path)
            
videofiles = [str for str in os.listdir(input_path) if any(sub in str for sub in formats)]
if len(videofiles) ==0:
    print('no video file found in',input_path) 
else:
    n=0
    m=0
    videototalframes = [imageio.get_reader(os.path.join(input_path,p)).count_frames() for p in videofiles ]
    videotargetframes = [math.ceil(f/sum(videototalframes)*target_frames) for f in videototalframes]
    intervals = [math.ceil(videototalframes[i]/videotargetframes[i]) for i in range(len(videototalframes))]
    totaloutputframe = sum([videototalframes[i]//intervals[i]+1 for i in range(len(videototalframes))])

    for i,videofile in enumerate(videofiles):
        path = os.path.join(input_path,videofile)
        reader = imageio.get_reader(path)
        r = reader._meta['size']
        #print(f'{videofile} no.of.frames:',reader.count_frames())
        m+=videotargetframes[i]
        
        #frames = [(i, j, im, n, target_res, res_ratio, r, output_path, m, totaloutputframe) for j,im in enumerate(reader)]
        #pool = multiprocessing.Pool()
        #pool.starmap(run_extract, frames)
        #pool.close()
        #pool.join
        
        n=0
        print('Extracting images from video...')
        with tqdm(total=videototalframes[i]) as pbar:
            
            for frame, im in enumerate(reader):
                if frame % intervals[i] == 0:
                    #print(frame)
                    n+=1
                    
                    if math.floor(target_res)==0:
                        target_res = r[0]*r[1]
                    rand = random.random()
                    if rand > res_ratio:
                        res = target_res
                    else:
                        res = target_res/2
                    y = math.sqrt(res/r[0]*r[1])
                    x = r[0]/r[1]*y
                    output_res = (int(x),int(y))
                    
                    im = Image.fromarray(im).resize(output_res)
                    output = os.path.join(output_path,f'frame_{i}'+str(n).zfill(3)+'.jpg')
                    #print('output frame:',os.path.basename(output),f'{r}downscale_to{output_res} | {n} of {m} of {totaloutputframe}')
                    imageio.imwrite(output, im)
                #n+=1
                pbar.update(1)
