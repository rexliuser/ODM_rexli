import tarfile, requests, json, time
import os, shutil
import datetime
import asyncio

from custom_util import post_processing

def aws_await(downloadlink, paths, args, time1):
    with open(os.path.join(paths.path_projectname,'waiting.log'),'w') as f:
        f.write(f'PID:{os.getpid()}\n')
    received = False
    timeout = 2700
    try_every = 10
    
    try:
    
        for _ in range(timeout//try_every):
            response = requests.get(downloadlink)
            with open(os.path.join(paths.path_projectname,'waiting.log'),'a') as f:
                f.write(f"{datetime.datetime.now()}:getting downloadlink -- {response.status_code}\n")
            if response.ok:
                #post processing
                export_path = os.path.join(paths.path_projectname,'exports')
                output_path = os.path.join(paths.path_odm,'output_models',args.projectname)
                zip_path = os.path.join(paths.path_projectname,'output_aws.zip')
                with open(zip_path, 'wb') as f:
                    f.write(response.content)
                shutil.unpack_archive(zip_path, os.path.join(paths.path_projectname,'unzip'))
                t = ''
                for folder in os.listdir(os.path.join(paths.path_projectname,'unzip')):
                    foldername = folder+'_aws'
                    try:
                        print('Removing folder: ', folder)
                        if os.path.exists(os.path.join(paths.path_projectname,foldername)):
                            shutil.rmtree(os.path.join(paths.path_projectname,foldername))
                    except:
                        print('do not have enough permission to delete odm_filterpoints, odm_texturing')
                        t = '_'
                    if os.path.isdir(os.path.join(paths.path_projectname,'unzip',folder)):
                        os.rename(os.path.join(paths.path_projectname,'unzip',folder),os.path.join(paths.path_projectname,foldername+t))
                os.rmdir(os.path.join(paths.path_projectname,'unzip'))
                #os.remove(zip_path)
                #if not os.path.exists(export_path):
                #    os.mkdir(export_path)
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                
                
                #with open(os.path.join(export_path,'mesh.glb'), 'wb') as f:
                #    f.write(response.content)
                #with open(os.path.join(output_path,'mesh.glb'), 'wb') as f:
                #    f.write(response.content)
                received = True
                
                break
            try:
                time.sleep(try_every)
            except:
                raise Exception("error occured")

        if received:
            post_processing.post_processing(paths, args)
            print('ODM with python script completed.')
            time2 = datetime.datetime.now()
            diff = time2-time1
            with open(os.path.join(paths.path_projectname,'waiting.log'),'a') as f:
                f.write(f"Total Time: {diff.seconds//60} minutes,{diff.seconds-diff.seconds//60*60} seconds")
            return 0
        else:
            with open(os.path.join(paths.path_projectname,'waiting.log'),'a') as f:
                f.write(f'error on aws. timeout')
            return 'error on aws. timeout' 
    except Exception as e:
        with open(os.path.join(paths.path_projectname,'waiting.log'),'a') as f:
            f.write(e)
        return 'error'