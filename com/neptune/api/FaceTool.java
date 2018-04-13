package com.neptune.api;

// image tool
import com.persist.util.tool.Face.ImageInfo;
import com.neptune.utils.FaceInfo;
import com.neptune.utils.FaceFeature;

import java.util.List;
import java.util.ArrayList;

/**
 * Created by YangFan on 17-07-03
 * 
 * detect face and verify similarity
 * 
 */

public class FaceTool {

    static {
        System.out.println("Loading shared lib: face");
        System.load("/home/yf/share/TestApi/newapi/build/libJniFace.so");
    }

    public native static boolean init(String config_path);

    public native static ArrayList<FaceInfo> detect(ImageInfo image);

    public native static ArrayList<FaceFeature> extract(ImageInfo image);

    public native static float verify(ImageInfo image1, ImageInfo image2);
}
