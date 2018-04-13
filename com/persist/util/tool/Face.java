package com.persist.util.tool;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.*;

/**
 * Created by YangFan 17-07-03
 *
 * Toy class for api development
 * 
 */
public class Face {
    
    public static class ImageInfo
    {
        public byte[] pixels;
        public int width;
        public int height;

        public ImageInfo()
        {

        }

        public ImageInfo(byte[] pixels, int width, int height)
        {
            this.pixels = pixels;
            this.width = width;
            this.height = height;
        }

        public ImageInfo(String imagePath) throws Exception {
            BufferedImage image = ImageIO.read(new FileInputStream(imagePath));
            this.pixels = ((DataBufferByte)image.getRaster().getDataBuffer()).getData();
            this.width = image.getWidth();
            this.height = image.getHeight();
        }

    }

    public static void main (String args[]) throws Exception {
        ImageInfo info = new ImageInfo(args[0]);
        System.out.println("width: " + info.width);
        System.out.println("height: " + info.height);
    }
}

