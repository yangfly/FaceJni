package com.neptune.utils;

public class FaceInfo {
	public BBox bbox;   // bounding box
	public float score; // confidence
	public FPoints fpts;  // facial landmarks

	public FaceInfo(float[] arr) {
		bbox = new BBox(subArray(arr, 0, 4));
		score = arr[4];
		fpts = new FPoints(subArray(arr, 5, 15));
	}

	public static float[] subArray(float[] src, int begin, int end) {
		float[] dest = new float[end-begin];
		System.arraycopy(src, begin, dest, 0, dest.length);
		return dest;
	}

	public class Point {
		public int x;
		public int y;

		public Point(float x, float y) {
			this.x = Math.round(x);
			this.y = Math.round(y);
		}
	}

	public class BBox {
		public Point lt;   // left top (x1, y1)
		public Point rd;   // right down (x2, y2)

		public BBox(float[] arr) {
			lt = new Point(arr[0], arr[1]);
			rd = new Point(arr[2], arr[3]);
		}
	}

	public class FPoints {
		public Point leye;   // left eye
		public Point reye;   // right eye
		public Point nose;   // nose
		public Point lmouth; // left mouth
		public Point rmouth; // right mouth

		public FPoints(float[] arr) {
			leye = new Point(arr[0], arr[1]);
            reye = new Point(arr[2], arr[3]);
            nose = new Point(arr[4], arr[5]);
            lmouth = new Point(arr[6], arr[7]);
            rmouth = new Point(arr[8], arr[9]);
		}
	}

	// public class Rotation {
	//     /* Todo: NotImplemented */
	//     public float roll;  // z axis  歪着头 
	//     public float pitch; // x axis  上下看
	//     public float yaw;   // y axis  左右看 

	//     public Rotation(float[] arr) {
	//         roll  = arr[0];
	//         pitch = arr[1];
	//         yaw   = arr[2];
	//     }
	// }

}
