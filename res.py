import argparse
import glob
import os
import os.path as osp
from multiprocessing import Pool


def resize_videos(vid_item, args):
    full_path, vid_path = vid_item

    vid_name = osp.basename(vid_path).replace('.webm', '.mp4')
    rel_dir = osp.dirname(vid_path)
    out_dir = osp.join(args.out_dir, rel_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_full_path = osp.join(out_dir, vid_name)

    if osp.exists(out_full_path):
        print(f"Skipping {vid_name}")
        return True

    try:
        result = os.popen(
            f'ffprobe -hide_banner -loglevel error -select_streams v:0 '
            f'-show_entries stream=width,height -of csv=p=0 "{full_path}"'
        )
        w, h = [int(d) for d in result.readline().rstrip().split(',')]
    except:
        print(f"Error reading {full_path}")
        return False

    # CRF 20 提升质量（原来是 23）
    if w > h:
        cmd = (f'ffmpeg -hide_banner -loglevel error -i "{full_path}" '
               f'-vf scale=-2:{args.scale} '
               f'-c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p '
               f'-an "{out_full_path}" -y')
    else:
        cmd = (f'ffmpeg -hide_banner -loglevel error -i "{full_path}" '
               f'-vf scale={args.scale}:-2 '
               f'-c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p '
               f'-an "{out_full_path}" -y')

    os.system(cmd)
    print(f"Converted: {vid_name}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description='SSV2 webm to mp4 converter')
    parser.add_argument('src_dir', type=str, help='source webm directory')
    parser.add_argument('out_dir', type=str, help='output mp4 directory')
    parser.add_argument('--scale', type=int, default=240, help='short side scale')
    parser.add_argument('--num-worker', type=int, default=8, help='parallel workers')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    vid_paths = glob.glob(osp.join(args.src_dir, '**/*.webm'), recursive=True)
    print(f"Found {len(vid_paths)} webm files")

    if len(vid_paths) == 0:
        print("No webm files found!")
        exit(1)

    vid_items = [(v, osp.relpath(v, args.src_dir)) for v in vid_paths]

    pool = Pool(args.num_worker)
    results = pool.starmap(resize_videos, [(item, args) for item in vid_items])
    pool.close()
    pool.join()

    print(f"Done: {sum(results)}/{len(vid_paths)}")