import os, subprocess, time

path_batch = '../projects/_batch'
projects = os.listdir(path_batch)

project_list = projects

def run_from_batch_folder():
    for project in project_list:
        path_project = f"../projects/batch_{project}"
        path_batch_project = os.path.join(path_batch,project)
        while os.path.exists(path_project):
            path_project = path_project + '_'
        path_dataset = os.path.join(path_project,'dataset')
        if os.path.exists(os.path.join(path_batch,'parameter.txt')):
            with open(os.path.join(path_batch,'parameter.txt'),'r') as f:
                parameter = f.read()
            parameters = parameter
        else:
            parameters = '-m'
        
        os.mkdir(path_project)
        os.mkdir(path_dataset)
        
        for file in os.listdir(path_batch_project):
            os.rename(os.path.join(path_batch_project,file),os.path.join(path_project,'dataset',file))
        
        print('Running project:',project)
        os.system(f'python run.py batch_{project} {parameters} > {path_project}/output_log.txt')
        os.rmdir(path_batch_project)

def simple_run():
    project_list = [pj for pj in sorted(os.listdir(f"../projects/")) if pj[:5]=='batch']
    for project in project_list:
        path_project = f"../projects/{project}"
        parameters = '-m -ap kill'
        print(f'|||||||||| Running {project} |||||||||||')
        #os.system(f'python run.py {project} {parameters} > output.log')
        subprocess.Popen(['python','run.py',project,parameters,'>','batch_output.log'])
        time.sleep(90)
    
simple_run()
    