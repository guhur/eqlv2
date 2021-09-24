import math
from typing import List, Tuple, Union, Set
from typing_extensions import Literal, TypedDict
import pickle
from pathlib import Path
import json
import argtyped
import lmdb
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from mmdet.apis import init_detector
from helper import inference_detector

Mode = Literal["bbox_feats", "roi_feats", "bboxes"]

class MatterportFeature(TypedDict):
    image_feat: List[Image.Image]
    view_ids: List[int]
    reverie_ids: List[int]
    image_h: int
    image_w: int
    fov: float
    boxes: List[List[int]]
    cls_probs: List[np.ndarray]


class Arguments(argtyped.Arguments):
    checkpoint: Path = Path("data/pretrain_models/eqlv2_1x.pth")
    config: Path = Path("configs/end2end/eqlv2_r50_8x2_1x.py")
    part_ids: List[int] = [0]
    num_parts: int = 1
    output: Path = Path("matterport_eqlv2.lmdb")
    matterport: Path = Path("matterport.lmdb")

    nms_thresh: float = 0.3
    conf_thresh: float = 0.4
    min_local_boxes: int = 5
    max_local_boxes: int = 20
    max_total_boxes: int = 100

    views_per_sweep: int = 12
    # number of total views from one pano
    viewpoint_size: int = 36
    heading_inc: int = 30
    # margin of error for deciding if an object is closer to the centre of another view
    angle_margin: int=5
    # elevation on first sweep
    elevation_start: int =-30 
    # how much elevation increases each sweep
    elevation_inc: int = 30 





def get_ft_head_elev(bbox: torch.Tensor, view_id: torch.Tensor, width: int, height: int, fov: float):
    # Calculate the heading and elevation of the center of each observation
    center_x = 0.5 * (bbox[:, 0] + bbox[:, 2])
    center_y = 0.5 * (bbox[:, 1] + bbox[:, 3])
    focal_length = torch.tensor((height / 2) / math.tan(math.radians(fov / 2))).to(bbox.device)

    heading = torch.deg2rad((view_id % 12) * 30)
    ft_heading = heading + torch.atan2(center_x - width / 2, focal_length)
    # normalize featureHeading
    ft_heading = ft_heading % (math.pi * 2)
    assert (0 <= ft_heading).all() and (ft_heading <= math.pi * 2).all()
    # force it to be the positive remainder, so that 0 <= angle < 360
    more_than_pi = ft_heading > math.pi
    ft_heading[more_than_pi] = (ft_heading - math.pi * 2)[more_than_pi]
    assert (-math.pi <= ft_heading).all() and (ft_heading <= math.pi).all()

    elevation = torch.deg2rad((view_id // 12) * 30 - 30)
    ft_elevation = elevation + torch.atan2(-center_y + height / 2, focal_length)

    return ft_heading, ft_elevation



def filter_panorama(boxes: torch.Tensor, probs: torch.Tensor, features: torch.Tensor, view_ids: torch.Tensor, max_boxes: int, width: int, height: int, fov: float) -> torch.Tensor:
    ft_heading, ft_elevation = get_ft_head_elev(boxes, view_ids, width, height, fov)
    # Remove the most redundant features (that have similar heading, elevation and 
    # are close together to an existing feature in cosine distance)
    pooled_features = features.sum(2).sum(2).unsqueeze(2)
    feat_dist = F.cosine_similarity(pooled_features, pooled_features.transpose(0, 2), dim=1)

    indices = torch.triu_indices(*feat_dist.shape, 1)

    heading_diff_tri = F.pdist(ft_heading.unsqueeze(1), 2)
    heading_diff_tri = torch.minimum(heading_diff_tri, 2*math.pi - heading_diff_tri)
    heading_diff = torch.zeros_like(feat_dist)
    heading_diff[indices[0], indices[1]] = heading_diff_tri

    elevation_diff_tri = F.pdist(ft_elevation.unsqueeze(1), 2)
    elevation_diff = torch.zeros_like(feat_dist)
    elevation_diff[indices[0], indices[1]] = elevation_diff_tri

    total_dist = feat_dist + heading_diff + elevation_diff # Could add weights

    # Discard diagonal and upper triangle by setting large distance
    dist = total_dist[indices[0], indices[1]]
    arg_ind = torch.argsort(dist)

    # Remove indices of the most similar features (in appearance and orientation)
    keep = set(range(feat_dist.shape[0]))
    ix = 0
    while len(keep) > max_boxes:
        min_ind = arg_ind[ix]
        i, j = indices[:, min_ind].tolist()

        if i not in keep or j not in keep:
            ix += 1
            continue

        if probs[i,1:].max() > probs[j,1:].max():
            keep.remove(j)
        else:
            keep.remove(i)
        ix += 1

    return torch.Tensor(list(keep)).long().to(boxes.device)


def extract_feat(worker_id, viewpoint_lists, args: Arguments):
    part_id = args.part_ids[worker_id]

    viewpoint_list = viewpoint_lists[part_id]
    num_viewpoints = len(viewpoint_list)
    print("Number of viewpoints on split{}: {}.".format(part_id, num_viewpoints))

    if args.num_parts == 1:
        lmdb_path = args.output
    else:
        lmdb_path = args.output.parent / f"{args.output.name}-{part_id}{args.output.suffix}"
    writer = LMDBWriter(str(lmdb_path), map_size=int(1e12), buffer_size=300)

    num_gpus = torch.cuda.device_count()
    assert num_gpus != 0
    device_id = part_id % num_gpus
    torch.cuda.set_device(device_id)
    model = init_detector(str(args.config), str(args.checkpoint), device=f'cuda:{device_id}')

    done = set(tuple(bkey.decode().split("_")) for bkey in  writer._keys)
    print('Done', len(done))
    viewpoint_list = [v for v in viewpoint_list if v not in done]
    print('Todo', len(viewpoint_list))

    dataset = MatterportDataset(viewpoint_list, args)

    disable = part_id != min(args.part_ids)
    for feats, scan, viewpoint in tqdm(dataset, disable=disable):
        all_boxes = []
        all_features = []
        all_probs = []
        all_view_ids = []
        all_labels = []

        for view_id, im in zip(feats['view_ids'], feats['image_feat']):

            results = inference_detector(model, np.array(im))
            assert len(results['bbox']) == len(results['features'])

            all_boxes.append(results['bbox'])
            all_probs.append(results['cls_score'])
            all_features.append(results['features'])
            all_labels.append(results['labels'])
            num_bbox = results['bbox'].shape[0]
            all_view_ids += [view_id] * num_bbox

        image_feat = torch.cat(all_features)
        bbox = torch.cat(all_boxes)
        probs = torch.cat(all_probs)
        view_ids = torch.Tensor(all_view_ids)
        labels = torch.cat(all_labels)

        keep_ind = filter_panorama(
            bbox,
            probs,
            image_feat,
            view_ids,
            args.max_total_boxes, 
            feats['image_w'],
            feats['image_h'],
            feats['fov'],
        )
        assert keep_ind.numel() > 0

        image_feat = image_feat[keep_ind]
        bbox = bbox[keep_ind]
        probs = probs[keep_ind]
        view_ids = view_ids[keep_ind]
        labels = labels[keep_ind]

        data = {
            "image_feat": image_feat,
            "boxes": bbox,
            "image_h": feats['image_h'],
            "image_w": feats['image_w'],
            "fov": feats['fov'],
            "view_ids": view_ids,
            "labels": labels,
            "cls_probs": probs,

        }

        key = f"{scan}_{viewpoint}"
        writer.put(key.encode('ascii'), pickle.dumps(data))
        writer.flush()




def load_json(path):
    with open(path, "r") as fid:
        data = json.load(fid)
    return data


class LMDBWriter:
    def __init__(self, path: Union[Path, str], map_size: int, buffer_size: int):
        self._env = lmdb.open(str(path), map_size=map_size, max_readers=512)
        self._buffer: List[Tuple[bytes, bytes]] = []
        self._buffer_size = buffer_size

        with self._env.begin(write=False) as txn:
            bkeys = txn.get("__keys__".encode("ascii"))
            if bkeys is None:
                self._keys: Set[bytes] = set()
            else:
                self._keys = set(pickle.loads(bkeys))

    def put(self, bkey: bytes, value: bytes):
        self._buffer.append((bkey, value))
        self._keys.add(bkey)
        if len(self._buffer) == self._buffer_size:
            self.flush()

    def get_keys(self) -> List[str]:
        return [k.decode("ascii") for k in self._keys]

    def flush(self):
        with self._env.begin(write=True) as txn:
            for bkey, value in self._buffer:
                txn.put(bkey, value)
            txn.put("__keys__".encode("ascii"), pickle.dumps(self._keys))

        self._buffer = []


def load_viewpoints(args: Arguments) -> List[Tuple[str, str]]:
    assert args.matterport.is_dir()
    env = lmdb.open(str(args.matterport))
    with env.begin(write=False) as txn:
        keys = txn.get('__keys__'.encode('ascii'))
        if keys is not None:
            keys = pickle.loads(keys)
        else:
            keys = []

    scan_viewpoints = []
    for key in keys:
        key = key.decode().split('_')
        scan_viewpoints.append((key[0], key[1]))

    return sorted(scan_viewpoints)


class MatterportDataset(Dataset):
    def __init__(self, viewpoints: List[Tuple[str, str]], args: Arguments):
        self.viewpoints = viewpoints
        self.args = args
        self.counter = 0
        self.env = lmdb.open(str(args.matterport))
        self.txn = self.env.begin(write=False)

    def __len__(self):
        return len(self.viewpoints)

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        self.counter += 1
        if self.counter == len(self):
            raise StopIteration()
        return self[self.counter]

    def __getitem__(self, index: int) -> Tuple[MatterportFeature, str, str]:
        scan, viewpoint = self.viewpoints[index]

        key = f"{scan}_{viewpoint}"
        feats = self.txn.get(key.encode('ascii'))
        if feats is None:
            raise RuntimeError()
        feats = pickle.loads(feats)

        return feats,  scan, viewpoint


def xywh2xyxy(bbox):
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]


def transform_img(im):
    """ Prep opencv BGR 3 channel image for the network """
    blob = np.array(im, copy=True)
    return blob


if __name__ == "__main__":
    args = Arguments()

    try:
         set_start_method('spawn', force=True)
    except RuntimeError:
        pass


    # Extract features.
    viewpoints = load_viewpoints(args)
    viewpoints_lists = [viewpoints[i::args.num_parts] for i in range(args.num_parts)]

    if args.part_ids == []:
        raise ValueError()
    elif len(args.part_ids) == 1:
        extract_feat(0, viewpoints_lists, args)
    else:
        parts = (viewpoints_lists, args)
        print('Spawn ', len(args.part_ids), 'procs')
        mp.spawn(extract_feat, nprocs=len(args.part_ids), args=parts)
