_base_ = ['./mask_rcnn_r50_8x2_1x.py']

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="EQLv2"))))
test_cfg = dict(rcnn=dict(max_per_img=100))
work_dir = 'eqlv2_1x'
