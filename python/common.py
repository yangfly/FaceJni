#!/usr/bin/env python
# coding=utf-8

import os, sys
caffe_root = '/home/yf/caffe-rc5'
sys.path.append(caffe_root + '/python')
os.environ['GLOG_minloglevel'] = '2'
import caffe
import numpy as np
import cv2
import math

# ---------------------------------
# bbox: x1, y1, x2, y2
# reg: dx1, dy1, dx2, dy2
# ---------------------------------

def roundint(x):
    return int(round(x))

def imread(filename):
    img = cv2.imread(filename)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    sample = img.astype(np.float32)
    return sample

def imsave(filename, img):
    cv2.imwrite(filename, img)

def imshow(img, bboxes, scores, fpts=None):
    # face bbox color
    green = (0, 255, 0)
    # face score color
    red = (0, 0, 255)
    for bbox, score in zip(bboxes, scores):
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), green)
        cv2.putText(img, '%.2f'%score, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_PLAIN, 1, red, thickness = 1)
    if fpts:
        for fpt in fpts:
            for i in range(5):
                cv2.circle(img, (int(fpt[2*i]), int(fpt[2*i+1])), 1, red)
    imsave('face.jpg', img)
    os.system('display face.jpg')
    os.remove('face.jpg')

def read_list(image_list):
    with open(image_list, 'r') as f:
        images = [line.strip() for line in f.readlines()]
        return images

def write_record(f, image, bboxes, scores):
    f.write('%s\n' % image)
    f.write('%d\n' % len(bboxes))
    for bbox, score in zip(bboxes, scores):
        record = []
        record.append(int(bbox[0]))  # x
        record.append(int(bbox[1]))  # y
        record.append(int(bbox[2] - bbox[0]))  # w
        record.append(int(bbox[3] - bbox[1]))    # h
        record.append(score)    # score
        f.write('%d %d %d %d %f\n' % tuple(record))

def set_batch_size(net, batch_size):
    data_name = net.inputs[0]
    data_shape = list(net.blobs[data_name].data.shape)
    data_shape[0] = batch_size
    net.blobs[data_name].reshape(*data_shape)

def scale_pyramid(height, width, factor, min_size):
    scales = []
    min_len = min(height, width)
    max_scale = 12. / min_size
    min_scale = 12. / min_len
    scale = max_scale
    while scale >= min_scale:
        scales.append(scale)
        scale *= factor
    return scales

def get_candidates(scale, out_prob, out_reg, thre):
    ''' x1 y1 x2 y2 score dx1 dy1 dx2 dy2 '''
    stride = 2
    cell_size = 12
    out_height, out_width = out_prob.shape
    # cv2.imwrite('heatmap_%.3f.png' %scale, out_prob*255)

    cands = []
    for i in range(out_height):
        for j in range(out_width):
            if out_prob[i, j] >= thre:
                cand = []
                cand.append(j * stride / scale)   # x1
                cand.append(i * stride / scale)   # y1
                cand.append((j * stride + cell_size - 1) / scale + 1)  # x2
                cand.append((i * stride + cell_size - 1) / scale + 1)  # y2
                cand.append(out_prob[i, j])   # score
                cand.append(out_reg[0, i, j])   # reg_x1
                cand.append(out_reg[1, i, j])   # reg_y1
                cand.append(out_reg[2, i, j])   # reg_x2
                cand.append(out_reg[3, i, j])   # reg_y2
                cands.append(cand)
    return cands

def non_max_supress(cands, thre, type):
    if len(cands) <= 1:
        return cands

    cands = sorted(cands, cmp=lambda x, y : cmp(x[4],y[4]), reverse=True)
    nms_cands = []
    while len(cands) > 0:
        max_cand = cands[0]
        max_area = (max_cand[2] - max_cand[0]) \
                 * (max_cand[3] - max_cand[1])
        cands.remove(max_cand)
        nms_cands.append(max_cand)

        idx = 0
        while idx < len(cands):
            # computer intersection
            x1 = max(max_cand[0], cands[idx][0])
            y1 = max(max_cand[1], cands[idx][1])
            x2 = min(max_cand[2], cands[idx][2])
            y2 = min(max_cand[3], cands[idx][3])
            overlap = None
            if (x1 < x2 and y1 < y2):
                inter = (x2 - x1) * (y2 - y1)
                area = (cands[idx][2] - cands[idx][0]) \
                     * (cands[idx][3] - cands[idx][1])
                outer = None
                if (type == 'Min'):
                    outer = min(max_area, area);
                else:
                    outer = max_area + area - inter
                overlap = float(inter) / outer
            else:
                overlap = 0

            if overlap > thre:
                cands.remove(cands[idx])
            else:
                idx += 1
    return nms_cands

def bbox_regress(cands):
    ''' x1 y1 x2 y2 score dx1 dy1 dx2 dy2 '''
    bboxes = []
    scores = []
    fpts = []
        
    for cand in cands:
        width = cand[2] - cand[0]
        height = cand[3] - cand[1]
        if len(cand) > 9:   # has fpts
            for i in range(9, 19, 2):
                cand[i] = cand[0] + cand[i] * width   # x
                cand[i+1] = cand[1] + cand[i+1] * height # y
        # bbox regression
        cand[0] += cand[5] * width  # x1
        cand[1] += cand[6] * height # y1
        cand[2] += cand[7] * width  # x2
        cand[3] += cand[8] * height # y2

def square_int(bboxes):
    '''convert rectangle float bboxes to square int bboxes'''
    sbboxes = []
    for bbox in bboxes:
        sbbox = []
        sbbox.append(int(math.floor(bbox[0])))    # x1
        sbbox.append(int(math.floor(bbox[1])))    # y1
        sbbox.append(int(math.ceil(bbox[2])))     # x2
        sbbox.append(int(math.ceil(bbox[3])))     # y2
        around_w = sbbox[2] - sbbox[0]
        around_h = sbbox[3] - sbbox[1]

        diff = around_w - around_h
        if diff > 0:    # width > height
            sbbox[1] -= diff / 2
            sbbox[3] += diff / 2
            if diff % 2 != 0:
                if (bbox[1] - sbbox[1]) < (sbbox[3] - bbox[3]):
                    sbbox[1] -= 1
                else:
                    sbbox[3] += 1
        elif diff < 0:  # width < height
            diff = -diff
            sbbox[0] -= diff / 2
            sbbox[2] += diff / 2
            if diff % 2 != 0:
                if (bbox[0] - sbbox[0]) < (sbbox[2] - bbox[2]):
                    sbbox[0] -= 1
                else:
                    sbbox[2] += 1
        sbboxes.append(sbbox)
    return sbboxes

# def test_square_int():
#     bboxes = [
#         [3.9, 4.6, 8.1, 9.05],
#         [3.9, 4.6, 8.1, 11.05],
#         [1.9, 4.6, 8.1, 9.05],
#         [3.9, 4.6, 8.1, 8.05],
#         [3.9, 4.6, 8.1, 8.9],
#         [3.9, 4.6, 7.2, 9.05],
#         [3.8, 4.6, 7.1, 9.05]
#     ]
#     sbboxes = square_int(bboxes)
#     assert sbboxes[0] == [3, 4, 9, 10]
#     assert sbboxes[1] == [2, 4, 10, 12]
#     assert sbboxes[2] == [1, 3, 9, 11]
#     assert sbboxes[3] == [3, 3, 9, 9]
#     assert sbboxes[4] == [3, 4, 9, 10]
#     assert sbboxes[5] == [3, 4, 9, 10]
#     assert sbboxes[6] == [2, 4, 8, 10]

def pad_crop(sample, sbbox):
    height, width = sample.shape[:2]
    inter_on_sample = [
        max(0, sbbox[0]),    # x1
        max(0, sbbox[1]),    # y1
        min(width, sbbox[2]),   # x2
        min(height, sbbox[3])]  # y2
    inter_on_crop = [
        inter_on_sample[0] - sbbox[0],
        inter_on_sample[1] - sbbox[1],
        inter_on_sample[2] - sbbox[0],
        inter_on_sample[3] - sbbox[1]]
    crop = np.zeros((sbbox[3] - sbbox[1], sbbox[2] - sbbox[0], 3), dtype=np.float32)
    crop[inter_on_crop[1]:inter_on_crop[3],
         inter_on_crop[0]:inter_on_crop[2], :] \
         = sample[inter_on_sample[1]:inter_on_sample[3],
                  inter_on_sample[0]:inter_on_sample[2], :]
    return crop

def Pnet(net, sample, thre, factor, min_size):
    sample_height, sample_width = sample.shape[:2]
    scales = scale_pyramid(sample_height, sample_width, factor, min_size)
    total_cands = []
    total_bboxes = []
    total_scores = []
    for scale in scales:
        height = int(math.ceil(sample_height * scale))
        width = int(math.ceil(sample_width * scale))
        img = cv2.resize(sample, (width, height))
        img = img.transpose((2, 0, 1))  # HWC -> CHW
        img = (img - 127.5) * 0.0078125 # nomalize

        set_batch_size(net, 1)
        net.blobs['data'].reshape(1, 3, height, width)
        net.blobs['data'].data[0, ...] = img
        out = net.forward()
        out_prob = out['prob'][0, -1, ...].copy()
        out_reg = out['reg'][0, ...].copy() 
        cands = get_candidates(scale, out_prob, out_reg, thre)
        # intra scale nms
        cands = non_max_supress(cands, 0.5, 'Union')
        if (len(cands) > 0):
            total_cands.extend(cands)
    # inter scale nms
    total_cands = non_max_supress(total_cands, 0.7, 'Union')
    # bbox regression
    bbox_regress(total_cands)

    bboxes = []
    scores = []
    for cand in total_cands:
        bboxes.append(cand[:4])
        scores.append(cand[4])

    return bboxes, scores

def Rnet(net, sample, thre, bboxes):
    num = len(bboxes)
    if num == 0:
        return [], []
    
    set_batch_size(net, num)
    data = np.zeros((num, 3, 24, 24), dtype=np.float32)
    sbboxes = square_int(bboxes)
    for i, sbbox in enumerate(sbboxes):
        crop = pad_crop(sample, sbbox)
        img = cv2.resize(crop, (24, 24))
        img = img.transpose((2, 0, 1))  # HWC -> CHW
        data[i, ...] = (img - 127.5) * 0.0078125 # nomalize
    # take care of gpu memory
    net.blobs['data'].data[...] = data
    out = net.forward()
    out_reg = out['reg'].copy()
    out_prob = out['prob'][:, -1].copy()
    cands = []
    for i in range(num):
        if out_prob[i] > thre:
            cand = []
            cand.extend(sbboxes[i])  # bbox
            cand.append(out_prob[i]) # score
            cand.extend(list(out_reg[i]))   # reg
            cands.append(cand)
    cands = non_max_supress(cands, 0.7, 'Union')
    # bbox regression
    bbox_regress(cands)

    bboxes = []
    scores = []
    for cand in cands:
        bboxes.append(cand[:4])
        scores.append(cand[4])

    return bboxes, scores

def Onet(net, sample, thre, bboxes):
    num = len(bboxes)
    if num == 0:
        return [], [], []

    set_batch_size(net, num)
    data = np.zeros((num, 3, 48, 48), dtype=np.float32)
    sbboxes = square_int(bboxes)
    for i, sbbox in enumerate(sbboxes):
        crop = pad_crop(sample, sbbox)
        img = cv2.resize(crop, (48, 48))
        img = img.transpose((2, 0, 1))  # HWC -> CHW
        data[i, ...] = (img - 127.5) * 0.0078125 # nomalize
    # take care of gpu memory
    net.blobs['data'].data[...] = data
    out = net.forward()
    out_reg = out['reg'].copy()
    out_prob = out['prob'][:, -1].copy()
    out_fpt = out['fpt'].copy()
    cands = []
    for i in range(num):
        if out_prob[i] > thre:
            cand = []
            cand.extend(sbboxes[i])  # bbox
            cand.append(out_prob[i]) # score
            cand.extend(list(out_reg[i])) # reg
            cand.extend(list(out_fpt[i])) # fpts
            cands.append(cand)

    bbox_regress(cands)
    cands = non_max_supress(cands, 0.7, 'Min')

    bboxes = []
    scores = []
    fpts = []
    for cand in cands:
        bboxes.append(cand[:4])
        scores.append(cand[4])
        fpts.append(cand[9:])

    return bboxes, scores, fpts

def Lnet(net, sample, bboxes, fpts):
    num = len(bboxes)
    if num == 0:
        return [], []

    set_batch_size(net, num)
    data = np.zeros((num, 15, 24, 24), dtype=np.float32)
    spatchses = np.zeros((num, 5), dtype=np.int)
    for i, pack in enumerate(zip(bboxes, fpts)):
        bbox, fpt = pack
        patchw = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        patchs = patchw * 0.25
        for j in range(5):
            patch = []
            patch.append(fpt[2*j+0] - patchs * 0.5)
            patch.append(fpt[2*j+1] - patchs * 0.5)
            patch.append(fpt[2*j+0] + patchs * 0.5)
            patch.append(fpt[2*j+1] + patchs * 0.5)
            spatch = square_int([patch])[0]
            spatchses[i,j] = spatch[2] - spatch[0]
            crop = pad_crop(sample, spatch)
            img = cv2.resize(crop, (24, 24))
            img = img.transpose((2, 0, 1))  # HWC -> CHW
            data[i, (3*j):(3*j+3), ...] = (img - 127.5) * 0.0078125 # nomalize
    # take care of gpu memory
    net.blobs['data'].data[...] = data
    out = net.forward()
    landmarks = ['lfeye', 'rteye', 'nose', 'lfmou', 'rtmou']
    for i, fpt in enumerate(fpts):
        for j, landmark in enumerate(landmarks):
            spatchs = spatchses[i, j]
            off_x = out[landmark][i, 0] - 0.5
            off_y = out[landmark][i, 1] - 0.5
            # Dot not make large movement with relative offset > 0.35
            if abs(off_x) > 0.35 or abs(off_y) > 0.35:
                continue
            fpt[2*j] += off_x * spatchs
            fpt[2*j+1] += off_y * spatchs

    return fpts
