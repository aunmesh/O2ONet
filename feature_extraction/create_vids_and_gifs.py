def download_yt_vid(id, output_folder):
    
    import os
    from pytube import YouTube
    output_path = os.path.join(output_folder, id)
    
    if os.path.exists(output_path):
        return
    
    else:
        
        url_prefix = 'https://www.youtube.com/watch?v='
        url = url_prefix + id
        YouTube(url).streams.get_highest_resolution().download(output_path=output_folder, filename=id+'.mp4')



def download_yt_vids(anno_file_path, output_folder):
    
    import pickle as pkl
    
    f = open(anno_file_path, 'rb')
    anno = pkl.load(f)
    f.close()
    
    yt_ids = [ item['metadata']['yt_id'] for item in anno]
    yt_ids = list(set(yt_ids))
    not_done = []
    from tqdm import tqdm as tqdm
    for id in tqdm(yt_ids):
        try:
            download_yt_vid(id, output_folder)
        except:
            print("Error for " + id)
            not_done.append(id)
    return not_done

# anno_path = '/workspace/work/O2ONet/data/annotation.pkl'
# youtube_saving_path = '/workspace/data/data_folder/o2o/youtube2'
# failed = download_yt_vids(anno_path, youtube_saving_path)

def prune_anno(anno_path, not_done):
    
    import pickle as pkl
    
    f = open(anno_path, 'rb')
    anno = pkl.load(f)
    f.close()
    
    pruned_anno = []
    
    for item in anno:
        yt_id = item['metadata']['yt_id']
        if yt_id not in not_done:
            pruned_anno.append(item)
    
    return pruned_anno

# pruned_anno = prune_anno(anno_path, failed)
# f = open('annotations_minus_unavailable_youtube_videos.pkl','wb')
# import pickle as pkl
# pkl.dump(pruned_anno, f)
# f.close()# %%

def extract_gif(frame_index, yt_id, youtube_folder, output_folder, window_size):
    
    import cv2
    import os

    filename = yt_id + '_' + str(frame_index) + '_' + str(window_size) + '.gif'
    saving_location = os.path.join(output_folder, filename)

    if os.path.exists(saving_location):
        return 1
    
    video_loc = os.path.join(youtube_folder, yt_id + '.mp4')
    vid = cv2.VideoCapture(video_loc)
    
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    initial = frame_index - window_size
    final = frame_index + window_size

    if initial<0 or final>=frame_count:
        print("Issue for ", frame_index, " ",yt_id)
        return 0

    frames = []

    for i in range(frame_count):

        success, frame = vid.read()
        if i >= initial and i <= final and success:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)


    import imageio

    fps = 4
    imageio.mimsave( saving_location, frames, fps=4)
    del vid
    return 1

def extract_gifs(anno_path, youtube_folder, output_folder, window_size):
    '''
    anno_path: Extracts all the annotations 
    '''

    import pickle as pkl
    
    f = open(anno_path, 'rb')
    anno = pkl.load(f)
    f.close()
    
    from tqdm import tqdm as tqdm
    failed = []

    for a in tqdm(anno):
        frame_index = int(a['metadata']['frame no.'])
        yt_id = a['metadata']['yt_id']
        ret = extract_gif(frame_index, yt_id, youtube_folder, output_folder, window_size)
        
        if not ret:
            failed.append(a)

    return failed

anno_path = '/workspace/work/O2ONet/data/annotations_minus_unavailable_yt_vids.pkl'
youtube_path = '/workspace/data/data_folder/o2o/youtube2'
output_folder = '/workspace/data/data_folder/o2o/gifs_11'
window_size = 5

failed = extract_gifs(anno_path, youtube_path, output_folder, window_size)