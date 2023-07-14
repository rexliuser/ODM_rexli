import os, shutil

def post_processing(paths, args):
    
    folder_path = 'odm_texturing_aws' if args.aws else 'odm_texturing'
    
    path_obj = os.path.join(paths.path_projectname,folder_path,'odm_textured_model_geo.obj')
    path_glb = os.path.join(paths.path_exports,'mesh.glb')
    if os.path.exists(path_obj):
        print('Converting obj to glb...')
        os.system('obj2gltf -i '+path_obj+' -o '+path_glb)

    if args.server:
        src = os.path.join(paths.path_exports,'mesh.glb')
        if not os.path.exists(os.path.join(paths.path_odm,'output_models')):
            os.mkdir(os.path.join(paths.path_odm,'output_models'))
        dest_dir = os.path.join(paths.path_odm,'output_models',args.server)
        dest_file = os.path.join(dest_dir,'mesh.glb')
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        if os.path.exists(src):
            shutil.copyfile(src,dest_file)
        print('mesh copied to output_models folder')