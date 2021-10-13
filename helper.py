import warnings
from mmdet.apis.inference import LoadImage
from mmdet.core import bbox2roi
from mmcv.ops import RoIAlign, RoIPool
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
import torch


def inference_detector(model, img, bbox=None):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    # forward the model
    with torch.no_grad():
        img = data['img'][0]
        img_metas = data['img_metas'][0]
        x = model.extract_feat(img)
        
        if bbox is None:
            proposal_list = model.rpn_head.simple_test_rpn(x, img_metas)
            rois = bbox2roi(proposal_list)
            cfg = model.roi_head.test_cfg
        else:
            rois = bbox2roi(bbox)
            cfg = None

        # cls_score: N, 1204 --> classification score for each bbox
        # bbox_pred: N, 4812 = (1204 - 1) * 4 --> bbox for each cat
        # bbox_feats: N, 256, 7, 7
        bbox_results = model.roi_head._bbox_forward(x, rois)
        img_shape = img_metas[0]['img_shape']
        scale_factor = img_metas[0]['scale_factor']
        det_bboxes, det_labels = model.roi_head.bbox_head.get_bboxes(
            rois,
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            img_shape,
            scale_factor,
            rescale=True,
            cfg=cfg)

    if bbox is not None:
        labels = det_labels.argmax(1).cpu()
        cls_score = bbox_results['cls_score'][:0, :-1].cpu()
        assert (labels == cls_score).all()
        return {
            'bbox': bboxes.cpu(),
            'labels': labels,
            'cls_score': cls_score,
            'features': bbox_results['bbox_feats'][:0, :-1].cpu(),
        }
    if det_labels.numel() == 0:
        return {
            'bbox': det_bboxes.cpu(),
            'labels': det_labels.cpu(),
            'cls_score': bbox_results['cls_score'][:0, :-1].cpu(),
            'features': bbox_results['bbox_feats'][:0, :-1].cpu(),
        }

    inds = det_labels[:, 0]

    return {
        'bbox': det_bboxes.cpu(),
        'labels': det_labels[:, 1].cpu(),
        'cls_score': bbox_results['cls_score'][inds, :-1].cpu(),
        'features': bbox_results['bbox_feats'][inds, :-1].cpu(),
    }
