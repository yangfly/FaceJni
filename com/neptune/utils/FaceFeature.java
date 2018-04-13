package com.neptune.utils;

import com.neptune.utils.FaceInfo;

public class FaceFeature extends FaceInfo {
    public float[] feature;

    public FaceFeature(float[] arr) {
        super(subArray(arr, 0, 15));
        this.feature = subArray(arr, 15, arr.length);
    }
}
