import cv2
#import matplotlib.pyplot as plt
import pandas as pd
import os, shutil
import numpy as np
#from sklearn.cluster import DBSCAN
#from sklearn.preprocessing import StandardScaler
import datetime
import argparse
from heapq import heappush, heappop
import asyncio

parser = argparse.ArgumentParser(
        prog='mask_atar.py',
        description='Use astar and clustering method to select masks'
    )
#parser.add_argument('projectname')
parser.add_argument('image_folder_path')
parser.add_argument('--rerun', dest='rerun', action='store_true', help='Force rerun mask. Default False')
args = parser.parse_args()

projectname = args.image_folder_path.split("/")[-2]
rerun = args.rerun
images_path = os.path.abspath(args.image_folder_path)
masks_folder = os.path.join(images_path,'masks')
time0 = datetime.datetime.now()

def labels_to_dict_of_lists2(df,labels):
    dict = {}
    for a, label in enumerate(labels):
        if str(label) != '-1':
            if str(label) not in dict:
                dict[str(label)] = []
            name =  f"{df.iloc[a]['frame']}:{str(df.iloc[a]['id'])}"
            dict[str(label)].append(name)
    return dict

def get_best_score_in_label(df,labels):
    dict = labels_to_dict_of_lists2(df,labels)
    long_lst = [dict[f] for f in dict]
    lst = [[p.split(':')[0] for p in dict[f]] for f in dict]
    best_score=0
    best_list = []
    for i,ls in enumerate(lst):
        s = get_score(ls,frame_list)
        if s>best_score:
            best_score=s
            best_list = long_lst[i]
        #print(best_score)
    return best_score, best_list

def get_score(random_list, given_list):
    set1 = set(random_list)
    set2 = set(given_list)
    #print(set1)
    #print(set2)
    score = len(set1.intersection(set2))
    #print('first: ',score)
    score -= 1*(len(random_list)-len(set1))
    score -= max(len(random_list)-len(given_list),0)
    #print('second: ',1*(len(random_list)-len(set1)))
    #print('third:',max(len(random_list)-len(given_list),0))
    #print('final: ',score)
    return score

def get_dist_to_center(df, frame, id):
    dff = df.loc[(df['id']==int(id)) & (df['frame']==frame)]
    x_center = dff.iloc[0]['crop_box_w']/2
    y_center = dff.iloc[0]['crop_box_h']/2 
    x = (dff.iloc[0]['bbox_x0']*2+dff.iloc[0]['bbox_w'])/2
    y = (dff.iloc[0]['bbox_y0']*2+dff.iloc[0]['bbox_h'])/2
    return np.sqrt((x-x_center)**2+(y-y_center)**2)

def get_mean_dist_to_center(df, lst):
    dist = []
    for ls in lst:
        frame,id = ls.split(':')
        ds = get_dist_to_center(df,frame,id)
        dist.append(ds)
    return np.mean(dist)
async def read_csv(filepath):
    df = pd.read_csv(filepath)
    frame = os.path.basename(os.path.dirname(filepath))
    df['frame']=frame
    return df
async def read_all_csv(filepaths):
    tasks = [read_csv(filepath) for filepath in filepaths]
    results = await asyncio.gather(*tasks)
    return results
def merge_csv(frame_list,filepath='data.csv'):
    df = pd.DataFrame()
    file_paths = [os.path.join(masks_folder,frame,'metadata.csv') for frame in frame_list]
    loop = asyncio.get_event_loop()
    dfs = loop.run_until_complete(read_all_csv(file_paths))
    for df2 in dfs:
        df = pd.concat([df,df2],axis=0)
    df.to_csv(filepath, index=False)
    return df

def get_vector_from_list(best_list, df):
    lst = []
    for ls in best_list:
        frame,id = ls.split(':')
        dff = df.loc[(df['id']==int(id)) & (df['frame']==frame)]
        id,a,b,c,d,area = dff.iloc[0]['id'],dff.iloc[0]['bbox_x0'],dff.iloc[0]['bbox_y0'],dff.iloc[0]['bbox_w'],dff.iloc[0]['bbox_h'],dff.iloc[0]['area']
        vector = (a,b,a+c,b+d,area)
        lst.append(vector)
    return lst

def filterData1(df):
    df = df[df['area'] > 100000]
    df = df[df['area'] < 1300000]
    #df = df[(df['bbox_x0'] > 106) & (df['bbox_y0'] > 520) & (df['bbox_x0']+df['bbox_w']< 954) & (df['bbox_y0']+df['bbox_h']< 1365)]
    #df = df[(df['point_input_x'] > 0) & (df['point_input_x'] < 1060) & (df['point_input_y'] > 520) & (df['point_input_y'] < 1365)]
    return df
def filterData2(df):
    df = df[df['area'] > 45000]
    df = df[df['area'] < 1700000]
    #df = df[(df['point_input_x'] > 0) & (df['point_input_x'] < 1060) & (df['point_input_y'] > 520) & (df['point_input_y'] < 1365)]
    return df

def getListFromDBCluster():
    df = merge_csv(frame_list)
    df = filterData1(df)
    vectors = [(int(frame[-3:]),int(frame[-3:]),a+c/2,b+d/2,area) for a,b,c,d,area,frame in zip(df['bbox_x0'],df['bbox_y0'],df['bbox_w'],df['bbox_h'],df['area'],df['frame'])]
    vectors = StandardScaler().fit_transform(vectors)
    X = np.array(vectors)
    
    lst = []
    tmp_list = []
    for i in np.arange(0.31,1.01,0.03).tolist():
        db = DBSCAN(eps=i, min_samples=2).fit(X)
        labels = db.labels_
        score, best_list = get_best_score_in_label(df,labels)
        if score>10:
            #print(i,len(best_list),score)
            #print(sorted(best_list))
            if tuple(best_list) not in tmp_list:
                tmp_list.append(tuple(best_list))
                lst.append([score,i,sorted(best_list)])
            
    lst = sorted(lst, key=lambda x:x[0], reverse=True)
    #best_i = lst[0][1]
    lst = [ls[2] for ls in lst]
    best_list=lst[0]
    return best_list

def getDf(frame_list, filepath='data.csv'):
    df = merge_csv(frame_list,filepath)
    df = filterData2(df)
    df['name']=df['frame']+':'+df['id'].astype(str)
    df.set_index('name',inplace=True)    
    df.to_csv(filepath,index=False)
    return df

def createStateSpaceGraph(df, output_path='stateSpaceGraph.txt',depth=1):
    graph={}
    layers = getLayersDict(df)
    graph = getGraphFromLayers(df,layers,depth)
    #with open(output_path,'w') as f:
    #    for key, value in graph.items():
    #        value = [f"{item[1].split(':')[0][-3:]}:{item[1].split(':')[1].zfill(2)}|{'{:.2f}'.format(round(item[0],2))}" for item in value if ':' in item[1]]
    #        if key not in ['start','end']:
    #            f.write(f"{key.split(':')[0]}:{key.split(':')[1].zfill(2)}: {value}\n")
    return graph
def getLayersDict(df):
    dict = {}
    for index, row in df.iterrows():
        if row['frame'] not in dict:
            dict[row['frame']]=[]
        dict[row['frame']].append(index)
    return dict
def getMotivationH(stateSpaceGraph):
    dict = {}
    for key in stateSpaceGraph:
        dict[key]=0
    return dict
def getGraphFromLayers(df, layers, depth=1):
    dict = {}
    dict['end']=[]
    keys = sorted(layers.keys())
    dict['start']=[(0,name) for name in layers[keys[0]]]
    for current, row in df.iterrows():
        dict[current]=[]
        next_frames = []
        frame_current = row['frame']
        frame_index = keys.index(frame_current)
        a = min(frame_index+1,len(keys))
        b = min(frame_index+1+depth,len(keys))
        next_frames = keys[a:b]
        #for i in range(1,1+depth):
        #    next_index = frame_index + i
        #    if next_index != len(keys)-1:
        #        next_frames.append(keys[next_index])
        #    else: break
        if next_frames:
            for next_frame in next_frames:
                dict[current]=dict[current]+[(getInverseScore(df,current,target),target) for target in layers[next_frame]]
        else: dict[current]=[(0,'end')]
        
        #if row['frame']!=keys[-1]:
        #    next_frame = keys[keys.index(row['frame'])+1]
        #    dict[index]=[(getInverseScore(df,index,name2),name2) for name2 in layers[next_frame]]
        #    if next_frame != keys[-1]:
        #        next_frame = keys[keys.index(next_frame)+1]
        #        dict[index]=dict[index]+[(getInverseScore(df,index,name2),name2) for name2 in layers[next_frame]]
        #        if next_frame != keys[-1]:
        #            next_frame = keys[keys.index(next_frame)+1]
        #            dict[index]=dict[index]+[(getInverseScore(df,index,name2),name2) for name2 in layers[next_frame]]
            
    return dict
def removeDuplicate(best_list):
    lst = [ls.split(':')[0] for ls in best_list]
    #print('1:',lst)
    dup_lst = [ls for ls in lst if lst.count(ls)>1]
    #print('2:',dup_lst)
    lst = [ls for ls in best_list if ls.split(':')[0] not in dup_lst]
    #print('3:',lst)
    return lst   
    
def getInverseScore(df,name1,name2):
    # by overlap area
    x1,y1,w1,h1 = df.loc[name1,'bbox_x0'],df.loc[name1,'bbox_y0'],df.loc[name1,'bbox_w'],df.loc[name1,'bbox_h']
    x2,y2,w2,h2 = df.loc[name2,'bbox_x0'],df.loc[name2,'bbox_y0'],df.loc[name2,'bbox_w'],df.loc[name2,'bbox_h']
    img_w, img_h = df.loc[name1,'crop_box_w'],df.loc[name1,'crop_box_h']
    area1=df.loc[name1,'area']
    area2=df.loc[name2,'area']
    a1 = (x1,y1,x1+w1,y1+h1)
    a2 = (x2,y2,x2+w2,y2+h2)
    rect1 = w1*h1
    rect2 = w2*h2
    centerX = img_w/2
    centerY = img_h/2
    area = getOverlapArea(a1,a2)

    #overlap_percentage = area/(rect1+rect2-area)
    overlap_percentage2 = min(area/rect1,area/rect2)
    penalty = 0.05
    edge_penalty = 1-np.where(x1>5,0,penalty)-np.where(x2>5,0,penalty)-np.where(img_w-x1-w1>5,0,penalty)-np.where(img_w-x2-w2>5,0,penalty)
    edge_penalty -= np.where(y1>5,0,penalty)+np.where(y2>5,0,penalty)+np.where(img_h-y1-h1>5,0,penalty)+np.where(img_h-y2-h2>5,0,penalty)
    #non_overlap_area_ratio = min((rect1-area)/(rect2-area+.001),(rect2-area)/(rect1-area+.001))
    #dist = (abs(x1+w1/2-centerX)+abs(y1+h1/2-centerY)+abs(x2+w2/2-centerX)+abs(y2+h2/2-centerY))/(centerX+centerY)
    #density_diff = abs(df.loc[name2,'area']/(w2*h2)-df.loc[name1,'area']/(w1*h1))
    #ratio_rect = min(rect1/rect2,rect2/rect1)
    ratio_area = min(area1/area2,area2/area1)
    #size_penalty = 1-(rect1+rect2-area)/(img_w*img_h)
    return 1-overlap_percentage2*edge_penalty*ratio_area
    return 1-overlap_percentage*ratio_area*size_penalty
def getOverlapArea(a,b):
    dx = max(0,min(a[2],b[2])-max(a[0],b[0]))
    dy = max(0,min(a[3],b[3])-max(a[1],b[1]))
    return dx*dy
def aStar(stateSpaceGraph, h, startState, goalState, iter=0): 
    if not goalState in stateSpaceGraph.keys():
        raise RuntimeError(f"goalState {goalState} does not exist")
    if not startState in stateSpaceGraph.keys():
        raise RuntimeError(f"startState {startState} does not exist")
    frontier = []
    heappush(frontier, (h[startState], [startState]))
    #print('Initial frontier:',list(frontier))
    n=0
    max_i = 0
    while frontier and n<500:
        #print()
        node = heappop(frontier)
        if len(node[1])>max_i:
            max_i=len(node[1])
            iter+=1
            #print(str(iter).zfill(2),str(n).zfill(4),[f"{nod.split(':')[0][-3:]}:{nod.split(':')[1].zfill(2)}" if ':' in nod else nod for nod in node[1] ],node[0])
            n=0
        #print('node:',node)
        if (node[1][-1]==goalState): return node[1]
        #print('Exploring:',node[1][-1],'at cost',node[0])
        #print(stateSpaceGraph[node[1][-1]])
        for child in stateSpaceGraph[node[1][-1]]:
            #print('child:',child)
            heappush(frontier, (node[0]+child[0]-h[node[1][-1]]+h[child[1]], node[1]+[child[1]]))
            #print(frontier)
        n+=1
        #print(list(frontier)); #input()
    else:
        #print('breakthrough')
        length =0
        for item in frontier:
            if len(item[1])>length:
                length=len(item[1])
        longest_lists = []
        for item in frontier:
            if len(item[1])==length:
                heappush(longest_lists,item)
        #print('longest_lists:',longest_lists)
        return longest_lists[0][1][:-1] + aStar(stateSpaceGraph, h, longest_lists[0][1][-2], goalState, iter-1)[1:]

def fillWithAstar(best_list, stateSpaceGraph, h):
    #start_list = aStar(stateSpaceGraph, h, startState, best_list[0]) if startState!=best_list[0] else []
    #print('start_list',start_list)
    final_list = []
    for i, name in enumerate(best_list[:-1]):
        #print()
        #print(i,name)
        start = name
        end = best_list[i+1]
        #print('>>>',start,end)
        astarlist = aStar(stateSpaceGraph, h, start, end)
        final_list.extend(astarlist)
    return list(dict.fromkeys(final_list))
async def write_img(key,img_lst):
    dest_path = os.path.join(images_path,f'{key}_mask.png')
    if len(img_lst)>1:
        for mask in img_lst:
            source_path = os.path.join(images_path,'masks',key,mask)
            im = cv2.imread(source_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            img = np.zeros(im.shape[:2])
            img = im+img
            
        img = np.where(img>0,255,0).astype(np.uint8)
        cv2.imwrite(dest_path,img)
        print('>0',key)
            
    elif len(img_lst)==0:
        source_path = os.path.join(images_path,f'{key}.jpg')
        im = cv2.imread(source_path)
        img = np.full(im.shape[:2],0).astype(np.uint8)
        cv2.imwrite(dest_path,img)
        print('=0',key)
    else:
        source_path = os.path.join(images_path,'masks',key,img_lst[0])
        shutil.copy(source_path,dest_path)
async def write_all_img(dict):
    tasks = [write_img(key,dict[key]) for key in dict]
    results = await asyncio.gather(*tasks)
    return results
def export_masks(best_list):
    dict = {}
    for frame in frame_list:
        dict[frame]=[]
    for frame in best_list:
        f_n,id = frame.split(':')
        dict[f_n].append(f'{id}.png')
    #print(dict)
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(write_all_img(dict))
    #for key in dict:
    #    write_img(key,dict[key])
            #print('exporting',key)
def mark_time():
    global time0
    diff = datetime.datetime.now()-time0
    time0 = datetime.datetime.now()
    return f'{round(diff.total_seconds()*1000)}ms'

def Astar(best_list):
    print('Extracting csv...')
    df = getDf(frame_list,os.path.join(masks_folder,'astar.csv'))
    print(mark_time())

    print('Creating stateSpaceGraphs')
    stateSpaceGraph_5 = createStateSpaceGraph(df,'stateSpaceGraph_5.txt',depth=5)
    stateSpaceGraph_3 = createStateSpaceGraph(df,'stateSpaceGraph_3.txt',depth=3)
    stateSpaceGraph = createStateSpaceGraph(df,'stateSpaceGraph.txt')

    #print(stateSpaceGraph['frame_0027:3'])
    print(mark_time())

    motivationH = getMotivationH(stateSpaceGraph)
    best_list = removeDuplicate(best_list)
    print('best_list:',best_list)
    
    print('Running fillWithAstar...')
    best_list = fillWithAstar(best_list, stateSpaceGraph_5, motivationH)
    print('5:',best_list)
    print()
    best_list = fillWithAstar(best_list, stateSpaceGraph_3, motivationH)
    print('3:',best_list)
    print()
    final_list = fillWithAstar(best_list, stateSpaceGraph, motivationH)[1:-1]
    print(mark_time())
    print('Final result:',final_list)
    return final_list    


time1 = datetime.datetime.now()


if not os.path.exists(masks_folder) or rerun:
    print('Generating masks...')
    os.system(f'python mask2former.py {images_path}')
    #os.system(f'python amg.py --checkpoint {os.path.join("../","segment_anything_model_checkpoints","sam_vit_h_4b8939.pth")} --model-type "vit_h" --input {images_path} --output {os.path.join(images_path,"masks")}')
else:
    print(f'Skipping mask generation as folder {masks_folder} already exist')

frame_list = sorted([f.split('.')[0] for f in os.listdir(images_path) if not os.path.isdir(os.path.join(images_path,f)) and f[-3:]=='jpg'])
#width = 1060
#height= 1885


#print('Running Clustering...')
#best_list = getListFromDBCluster()
best_list = ['start','end']

print('Running Astar...')
final_list = Astar(best_list)

print('Exporting masks...')
export_masks(final_list)
print(mark_time())

time2 = datetime.datetime.now()
diff = time2-time1
print(f"Total Time: {diff.total_seconds()*1000}ms")

#if __name__ == "__main__":
#main()