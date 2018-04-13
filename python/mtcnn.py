#!/usr/bin/env python
# coding=utf-8

from common import *

class Detector :
    def __init__(self, protos, models, thres, factor, min_size):
        caffe.set_mode_gpu()
        caffe.set_device(0)
        self.nets = []
        assert len(protos) > 0 and len(protos) <= 4
        assert len(protos) == len(models), 'Mismatch num of protos and models'
        for proto, model in zip(protos, models):
            self.nets.append(caffe.Net(proto, model, caffe.TEST))
        self.thres = thres
        self.factor = factor
        self.min_size = min_size

    def set_last_thre(self, thre):
        self.thres[-1] = thre

    def last_thre(self):
        return self.thres[-1]

    def detect(self, img):
        bboxes, scores = Pnet(self.nets[0], img, self.thres[0], self.factor, self.min_size)
        if len(self.nets) == 1:
            return bboxes, scores
        bboxes, scores = Rnet(self.nets[1], img, self.thres[1], bboxes)
        if len(self.nets) == 2:
            return bboxes, scores
        bboxes, scores, fpts = Onet(self.nets[2], img, self.thres[2], bboxes)
        if len(self.nets) == 3:
            return bboxes, scores, fpts
        fpts = Lnet(self.nets[3], img, bboxes, fpts)
        if len(self.nets) == 4:
            return bboxes, scores, fpts
            
# test settings
model_dir = "../additions/mtcnn"
factor = 0.709
min_size = 40

def test():
    protos = [
        model_dir + '/det1.prototxt',
        model_dir + '/det2.prototxt',
        model_dir + '/det3.prototxt',
        model_dir + '/det4.prototxt']
    models = [
        model_dir + '/det1.caffemodel',
        model_dir + '/det2.caffemodel',
        model_dir + '/det3.caffemodel',
        model_dir + '/det4.caffemodel']
    thres = [0.5, 0.6, 0.6]
    detector = Detector(protos, models, thres, factor, min_size)
    img = imread("../test/test.jpg")
    bboxes, scores, fpts = detector.detect(img)
    print bboxes
    print scores
    print fpts
    imshow(img, bboxes, scores, fpts)

if __name__ == '__main__':
    test()
