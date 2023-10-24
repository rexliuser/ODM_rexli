# Running ODM with python scripts
import argparse, os, shutil, subprocess
import datetime
import tarfile, requests, json, time, multiprocessing
import asyncio

from custom_util import aws_await, post_processing, utils

parser = argparse.ArgumentParser(
    prog='run.py',
    description='Run ODM with python scripts'
)
parser.add_argument('projectname')
parser.add_argument('-m','--mask', dest='mask', action='store_true', help='use mask [True, False*]')
parser.add_argument('-m2','--mask2', dest='mask2', action='store_true', help='use Mask2Former (experimental) [True, False*]')
parser.add_argument('-s','--server',dest='server', type=str, help='timestamp from server. Toggle server mode. Default None', default=None)
parser.add_argument('-t','--test',dest='test', action='store_true', help='toggle test, override other settings [True, False*]')
parser.add_argument('-f','--frames', dest='frames', help='no. of frames from video. Default 70', default=70)
parser.add_argument('-r','--resolution', dest='resolution', help='target resolution of frames in MP, use 0 as original. Default 2',default=2)
parser.add_argument('-o','--outputmesh', dest='outputmesh', help='output mesh no. of face. Default 15000', default=15000)
parser.add_argument('-pq','--pc-quality', dest='pc_quality', help='pc-quality [ultra, high, medium*, low, lowest].', default='medium')
parser.add_argument('-fq','--feature-quality', dest='feature_quality', help='feature-quality [ultra*, high, medium, low, lowest]', default='ultra')
parser.add_argument('--norerun', dest='norerun', action='store_true', help='do not rerun pipeline. Default False (rerun)')
parser.add_argument('-sr',dest='sr', action='store_true', help='run super-resolution to x4', )
parser.add_argument('-ot',dest='octree', help='octree_depth default 7', default=7)
parser.add_argument('-kf',dest='keep_unseen_face', action='store_true', help='keep-unseen-face detault false')
parser.add_argument('-ap','--additional-parameters',dest='additional',help='additional parameters as string',default='')
parser.add_argument('--override',dest='override',help='custom parameters, override all',default='')
parser.add_argument('--aws',dest='aws',action='store_true',help='use aws fargate to do processing')
parser.add_argument('-pp',dest='preprocess',action='store_true',help='Preprocess only. Do not run ODM')
parser.add_argument('--start',dest='start', help='starting phase. (dataset|opensfm|openmvs|odm_filterpoints|odm_meshing|mvs_texturing|odm_report). Default ""', default='')

args = parser.parse_args()

projectname=args.projectname
mask = args.mask
#video_frames = args.videoframes
key = args.server
server = True if key is not None else False
test = args.test
frame_no = args.frames
mesh_size = args.outputmesh
pc_quality = args.pc_quality
feature_quality = args.feature_quality
norerun = args.norerun
sr = args.sr
resolution = float(args.resolution) if not sr else 0
mesh_octree_depth = args.octree
keep_unseen_face = args.keep_unseen_face
additional_parameters = args.additional
override = args.override
use_aws = args.aws
ti = '-ti' if not server else ''

def send_to_aws(commandlist, paths):
    print('------------------------')
    print('|  use AWS to process  |')
    print('------------------------')
    print('Uploading to aws...')
    with open(os.path.join(paths.path_images,'container_command.txt'),'w') as f:
        f.write(' '.join(commandlist[1:]))
    
    #bundle files to archive.tar
    filename_list = [os.path.join(paths.path_images,p) for p in os.listdir(paths.path_images) if not os.path.isdir(os.path.join(paths.path_images,p))]
    tarname = 'archive.tar'
    tarpath = os.path.join(paths.path_images,tarname)
    with tarfile.open(tarpath, 'w') as tar:
        for file in filename_list:
            tar.add(file, arcname=os.path.basename(file))
    
    #post to aws and receive s3 link for upload file and future download url
    url = 'aws_s3_upload_link'
    print('sending post to aws')
    response = requests.post(url)
    print('links received from aws')
    message = json.loads(json.loads(response.content.decode('utf-8')))
    upload_link = message['upload_link']
    download_link = message['download_link']
    
    #upload tar to s3
    with open(tarpath, 'rb') as f:
        files = {'file':(tarpath, f)}
        http_response = requests.post(upload_link['url'], data=upload_link['fields'], files=files)
    print('files sent to aws')
    os.remove(tarpath)
    with open(os.path.join(paths.path_projectname,'downloadlink.txt'),'w') as f:
        f.write(download_link)
    #process = multiprocessing.Process(target=aws_await.aws_await, args=(download_link, paths, args, time1))
    #process.daemon = True
    #process.start()
    aws_await.aws_await(download_link, paths, args, time1)
    print('Background process started: waiting for aws completion.')

    

class Paths:
    def __init__(self, projectname):
        self.projectname = projectname
        self.path_script = os.path.dirname(os.path.realpath(__file__))
        self.path_odm = os.path.dirname(self.path_script)
        self.path_projects = os.path.join(self.path_odm,'projects')
        self.path_projectname = os.path.join(self.path_projects,projectname)
        self.path_exports = os.path.join(self.path_projectname,'exports')
        self.path_dataset = os.path.join(self.path_projectname,'dataset')
        self.path_images =  os.path.join(self.path_projectname,'images')

def run_project():
    print(projectname, mask, server, test, frame_no, resolution, mesh_size, pc_quality, feature_quality,norerun)

    paths = Paths(args.projectname)

    if not os.path.exists(paths.path_images):
        os.mkdir(paths.path_images)
    if not os.path.exists(paths.path_exports):
        os.mkdir(paths.path_exports)
    if not os.path.exists(os.path.join(paths.path_exports,'refined')):
        os.mkdir(os.path.join(paths.path_exports,'refined'))

    #########
    if not norerun and args.start=='':
        reset_files = ['images.json','benchmark.txt','cameras.json','img_list.txt','log.json']
        for file in reset_files:
            filepath = os.path.join(paths.path_projectname,file)
            if os.path.exists(filepath):
                os.remove(filepath)
                print('Removed',file)
        try:
            print('Extracting frames from videos...')
            os.system('python videoToImages.py '+paths.path_dataset+' --o '+paths.path_images+' --f '+str(frame_no) + ' --r '+str(resolution))
        except Exception as e:
            raise RuntimeError(e)
        try:
            print('Downscaling images...')
            os.system('python downscale_img.py '+ paths.path_dataset+' -o '+ paths.path_images+' -r '+str(resolution))
        except Exception as e:
            raise RuntimeError(e)
        
        try:
            print('generating split...')
            utils.generate_split(paths.path_images, paths.path_projectname)
        except Exception as e:
            raise RuntimeError(e)
        
        try:
            if args.mask2:
                os.system(f'python mask_astar.py {paths.path_images} --rerun')
            elif mask:
                print('Applying mask...')
                os.system('python removeBG_parallel.py '+paths.path_images+' '+paths.path_images)
        except Exception as e:
            raise RuntimeError(e)
    
        if sr:
            print('Apply super-resolution:',)
            sr_path = os.path.join(paths.path_odm,'Real-ESRGAN','inference_realesrgan.py')
            commands = 'python '+sr_path+' -n RealESRGAN_x4plus -i '+paths.path_images+' -o '+paths.path_images+' -s 2'
            os.system(commands)
            
            for f in os.listdir(paths.path_images):
                if f.split('.')[0][-3:] != 'out':
                    os.remove(os.path.join(paths.path_images,f))
                    print('Removed: ',os.path.join(paths.path_images,f))

    print("Preprocessing completed.")
    
    if not args.preprocess:
        
        #video_only = True if len(os.listdir(paths.path_dataset))==1 else False
        seq_matching = True if len([f for f in os.listdir(paths.path_dataset) if not os.path.isdir(os.path.join(paths.path_dataset,f))])==1 else False
    
        projectinit = paths.path_projects+':/projects'
        #containerImage = 'opendronemap/odm:gpu' #'odm_mdai:latest'
        containerImage = 'rexliuser/odm_mdai_gpu:latest'
        #containerImage = 'opendronemap/odm:gpu'
        #containerImage = 'odm_mdai:latest'
        #'--user $(id -u):$(id -g)'
        commandlist = ['docker','run','--gpus','all',ti,'--rm','-v',projectinit,containerImage] #chown -R $(id -un):$(id -un) /projects
    
        commandlist2 = []
        if override == '':
            commandlist2=[
                projectname,
                '--project-path','/projects',
                '--feature-quality',feature_quality,
                '--matcher-type','flann', #flann
                '--pc-quality',pc_quality,
                '--mesh-size',str(mesh_size),
                '--skip-orthophoto',
                #'--skip-report',
                '--texturing-single-material',
                '--mesh-octree-depth',str(mesh_octree_depth),
                '--rerun-all'
                ]
            
            if seq_matching: commandlist2.extend(['--matcher-order-neighbors','7'])
            #if not norerun and args.start=='': commandlist2.extend(['--rerun-from',args.start])
            if keep_unseen_face: commandlist2.append('--texturing-keep-unseen-faces')
            if additional_parameters!='': commandlist2.append(additional_parameters)
        else:
            commandlist2=override.split(' ')
        
        if use_aws:
            send_to_aws(commandlist2, paths)
            
        else:
            command = ' '.join(commandlist+commandlist2)
            #command = command + f';/bin/sh -c "chown -R $(id -un):$(id -un) {paths.path_projectname}"'
            print('Running ODM Pipeline...')
            print('command:', command)
            

            result = os.system(command)
            
            post_processing.post_processing(paths, args)

            print('ODM with python script completed.')
            print('Return code:',result)

time1 = datetime.datetime.now()
run_project()
time2 = datetime.datetime.now()
diff = time2-time1
print(f'Project {args.projectname} completed!')
print(f"Total Time: {diff.seconds//60} minutes,{diff.seconds-diff.seconds//60*60} seconds")
