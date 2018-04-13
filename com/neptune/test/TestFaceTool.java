package com.neptune.test;

// image tools
import com.persist.util.tool.Face.ImageInfo;
import com.neptune.utils.FaceInfo;
import com.neptune.utils.FaceFeature;

import com.neptune.api.FaceTool;

import java.util.ArrayList;

public class TestFaceTool {

    public static void printFaceInfo(FaceInfo info) {
        System.out.println("[detect] "
                + info.bbox.lt.x + " " + info.bbox.lt.y + " "
                + info.bbox.rd.x + " " + info.bbox.rd.y + " "
                + info.fpts.leye.x + " " + info.fpts.leye.y + " "
                + info.fpts.reye.x + " " + info.fpts.reye.y + " "
                + info.fpts.nose.x + " " + info.fpts.nose.y  + " "
                + info.fpts.lmouth.x + " " + info.fpts.lmouth.y + " "
                + info.fpts.rmouth.x + " " + info.fpts.rmouth.y + " "
                + info.score);
    }

    public static void printFaceFeature(FaceFeature feat) {
        System.out.println("[extract] "
                + feat.bbox.lt.x + " " + feat.bbox.lt.y + " "
                + feat.bbox.rd.x + " " + feat.bbox.rd.y + " "
                + feat.score + " "
                + feat.fpts.leye.x + " " + feat.fpts.leye.y + " "
                + feat.fpts.reye.x + " " + feat.fpts.reye.y + " "
                + feat.fpts.nose.x + " " + feat.fpts.nose.y  + " "
                + feat.fpts.lmouth.x + " " + feat.fpts.lmouth.y + " "
                + feat.fpts.rmouth.x + " " + feat.fpts.rmouth.y + " "
                + feat.feature[0]);
    }

    public static void test_detect() throws Exception {
        System.out.println("--------- test detect ----------");
        ImageInfo image = new ImageInfo("test/test2.jpg");
        long start = System.currentTimeMillis();
        ArrayList<FaceInfo> infos = FaceTool.detect(image);
        System.out.println("detect use: " + (System.currentTimeMillis() - start));
        for (FaceInfo info : infos) {
            printFaceInfo(info);
        }
    }

    public static void test_verify() throws Exception {
        System.out.println("--------- test verify ----------");
        ImageInfo image1 = new ImageInfo("test/cdy_cdy_0_01.jpg");
        ImageInfo image2 = new ImageInfo("test/cdy_cdy_0_02.jpg");
        long start = System.currentTimeMillis();
        System.out.println("similarity: " + FaceTool.verify(image1, image2));
        System.out.println("verify use: " + (System.currentTimeMillis() - start));
    }

    public static void test_search() throws Exception {
        System.out.println("--------- test search ----------");
        ImageInfo image = new ImageInfo("test/test2.jpg");
        long start = System.currentTimeMillis();
        ArrayList<FaceFeature> feats = FaceTool.extract(image);
        System.out.println("search use: " + (System.currentTimeMillis() - start));
        for (FaceFeature feat: feats) {
            printFaceFeature(feat);
        }
    }

    public static void main(String args[]) throws Exception {
        if (FaceTool.init("config.json"))
            System.out.println("Init inference engine successfully.");
        else
            System.out.println("Failed to init inference engine.");
        test_detect();
        test_verify();
        test_search();
    }

}
