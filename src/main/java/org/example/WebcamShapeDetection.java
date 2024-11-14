package org.example;

import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.ArrayList;
import java.util.List;

public class WebcamShapeDetection {
    static {
        nu.pattern.OpenCV.loadShared();
    }

    private static final double EXPECTED_LENGTH_WIDTH_RATIO = 2.33; // 3.5/1.5
    private static final double RATIO_TOLERANCE = 0.3;
    private static final double KNOWN_WIDTH = 3.5; // Real width of object in inches
    private static final double FOCAL_LENGTH = 650.0; // This needs to be calibrated for your webcam

    private static Point[] lastDetectedPoints = null;
    private static int stabilityCounter = 0;
    private static final int STABILITY_THRESHOLD = 2;
    private static final double MOVEMENT_THRESHOLD = 40.0;

    private static final Scalar HSV_LOW = new Scalar(15, 100, 100);
    private static final Scalar HSV_HIGH = new Scalar(30, 255, 255);

    private static double calculateDistance(double pixelWidth) {
        // Using the formula: distance = (known width * focal length) / pixel width
        return (KNOWN_WIDTH * FOCAL_LENGTH) / pixelWidth;
    }

    private static double getPixelWidth(Point[] points) {
        // Calculate width as the minimum of the two parallel sides
        double width1 = Math.sqrt(Math.pow(points[1].x - points[0].x, 2) +
                Math.pow(points[1].y - points[0].y, 2));
        double width2 = Math.sqrt(Math.pow(points[2].x - points[3].x, 2) +
                Math.pow(points[2].y - points[3].y, 2));
        return Math.min(width1, width2);
    }

    public static void main(String[] args) {
        JFrame window = new JFrame("Webcam Shape Detection");
        JLabel screen = new JLabel();
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        window.add(screen);
        window.setSize(640, 480);
        window.setVisible(true);

        VideoCapture camera = new VideoCapture(0);
        camera.set(Videoio.CAP_PROP_FRAME_WIDTH, 640);
        camera.set(Videoio.CAP_PROP_FRAME_HEIGHT, 480);
        camera.set(Videoio.CAP_PROP_AUTO_EXPOSURE, 0.75);
        camera.set(Videoio.CAP_PROP_BRIGHTNESS, 0.5);

        Mat frame = new Mat();
        while (true) {
            if (camera.read(frame)) {
                processFrame(frame);
                Image image = matToBufferedImage(frame);
                screen.setIcon(new ImageIcon(image));
                window.repaint();
            }
        }
    }

    private static void processFrame(Mat frame) {
        Imgproc.GaussianBlur(frame, frame, new Size(3, 3), 0);

        Mat hsv = new Mat();
        Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_BGR2HSV);

        Mat colorMask = new Mat();
        Core.inRange(hsv, HSV_LOW, HSV_HIGH, colorMask);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3));
        Imgproc.morphologyEx(colorMask, colorMask, Imgproc.MORPH_OPEN, kernel);
        Imgproc.morphologyEx(colorMask, colorMask, Imgproc.MORPH_CLOSE, kernel);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(colorMask, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        List<Point[]> validObjects = new ArrayList<>();

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area < 1000) continue;

            MatOfPoint2f curve = new MatOfPoint2f(contour.toArray());
            MatOfPoint2f approxCurve = new MatOfPoint2f();
            double epsilon = 0.02 * Imgproc.arcLength(curve, true);
            Imgproc.approxPolyDP(curve, approxCurve, epsilon, true);

            if (approxCurve.total() >= 4 && approxCurve.total() <= 6) {
                RotatedRect rect = Imgproc.minAreaRect(curve);
                double width = rect.size.width;
                double length = rect.size.height;
                if (width > length) {
                    double temp = width;
                    width = length;
                    length = temp;
                }

                double ratio = length / width;
                if (Math.abs(ratio - EXPECTED_LENGTH_WIDTH_RATIO) <= RATIO_TOLERANCE) {
                    Point[] rectPoints = new Point[4];
                    rect.points(rectPoints);
                    validObjects.add(rectPoints);
                }
            }
        }

        for (Point[] points : validObjects) {
            // Draw rectangle
            for (int i = 0; i < 4; i++) {
                Imgproc.line(frame, points[i], points[(i + 1) % 4],
                        new Scalar(0, 255, 0), 2);
            }

            // Calculate and display distance
            double pixelWidth = getPixelWidth(points);
            double distance = calculateDistance(pixelWidth);

            Point center = new Point(
                    (points[0].x + points[1].x + points[2].x + points[3].x) / 4,
                    (points[0].y + points[1].y + points[2].y + points[3].y) / 4
            );

            Imgproc.putText(frame,
                    String.format("%.2f inches", distance),
                    center,
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    new Scalar(0, 255, 0),
                    1
            );
        }

        Imgproc.putText(frame, "Objects found: " + validObjects.size(), new Point(10, 20),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0), 1);

        hsv.release();
        colorMask.release();
    }

    private static boolean isStableDetection(Point[] current, Point[] last) {
        if (current.length != last.length) return false;

        for (int i = 0; i < current.length; i++) {
            double distance = Math.sqrt(
                    Math.pow(current[i].x - last[i].x, 2) +
                            Math.pow(current[i].y - last[i].y, 2)
            );
            if (distance > MOVEMENT_THRESHOLD) return false;
        }
        return true;
    }

    private static BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (mat.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        mat.get(0, 0, ((DataBufferByte) image.getRaster().getDataBuffer()).getData());
        return image;
    }
}