package com.example.smartglasses;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class ObjectDetectorClass {

    private final Interpreter interpreter;

    private final List<String> labels;
    private final int INPUT_SIZE;
    final float objThresh=0.7f;

    protected static final int BATCH_SIZE = 1;
    protected static final int PIXEL_SIZE = 3;

    ByteBuffer outData;

    ObjectDetectorClass(AssetManager assetManager, String modelPath, String labelPath, int inputSize) throws IOException {
        INPUT_SIZE = inputSize;

        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);

        //loading model
        interpreter = new Interpreter(loadModelFile(assetManager, modelPath), options);
        // load labelmap
        labels = loadLabelList(assetManager, labelPath);
    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        while((line=reader.readLine()) != null){
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        // use to get description of file
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    public Mat recognizeImage(Mat mat_image){

        int output_box = (int) ((Math.pow((INPUT_SIZE / 32), 2) + Math.pow((INPUT_SIZE / 16), 2) + Math.pow((INPUT_SIZE / 8), 2)) * 3);


        Bitmap bitmap;
        bitmap=Bitmap.createBitmap(mat_image.cols(),mat_image.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat_image,bitmap);
        // define height and width
        float height = bitmap.getHeight();
        float width = bitmap.getWidth();

        // scale the bitmap to input size of model
        Bitmap scaledBitmap=Bitmap.createScaledBitmap(bitmap,INPUT_SIZE,INPUT_SIZE,false);

        // convert bitmap to bytebuffer as model input should be in it
        ByteBuffer byteBuffer2=convertBitmapToByteBuffer(scaledBitmap);

        // defining output
        //  float[][][]result=new float[1][10][4];
        Object[] input=new Object[1];
        input[0]=byteBuffer2;

        Map<Integer, Object> output_map = new HashMap<>();

        int numClass = labels.size();
        outData = ByteBuffer.allocateDirect(output_box * (numClass + 5) * 4);
        outData.order(ByteOrder.nativeOrder());
        outData.rewind();
        output_map.put(0, outData);


        // predict
        interpreter.runForMultipleInputsOutputs(input,output_map);


        ByteBuffer byteBuffer = (ByteBuffer) output_map.get(0);
        byteBuffer.rewind();

        ArrayList<Recognition> detections = new ArrayList<>();

        float[][][] out = new float[1][output_box][numClass + 5];
        Log.d("YoloV5Classifier", "out[0] detect start");
        for (int i = 0; i < output_box; ++i) {
            for (int j = 0; j < 15 + 5; ++j) {
                out[0][i][j] = byteBuffer.getFloat();
            }
            // Denormalize x y w h
            for (int j = 0; j < 4; ++j) {
                out[0][i][j] *= INPUT_SIZE;
            }
        }
        for (int i = 0; i < output_box; ++i){
            final int offset = 0;
            final float confidence = out[0][i][4];
            int detectedClass = -1;
            float maxClass = 0;

            final float[] classes = new float[labels.size()];
            for (int c = 0; c < labels.size(); ++c) {
                classes[c] = out[0][i][5 + c];
            }

            for (int c = 0; c < labels.size(); ++c) {
                if (classes[c] > maxClass) {
                    detectedClass = c;
                    maxClass = classes[c];
                }
            }

            final float confidenceInClass = maxClass * confidence;
            if (confidenceInClass > objThresh) {
                final float xPos = out[0][i][0];
                final float yPos = out[0][i][1];

                final float w = out[0][i][2];
                final float h = out[0][i][3];
                Log.d("YoloV5Classifier",
                        Float.toString(xPos) + ',' + yPos + ',' + w + ',' + h);

//                final RectF rect =
//                        new RectF(
//                                Math.max(0, xPos - w / 2),
//                                Math.max(0, yPos - h / 2),
//                                Math.min(bitmap.getWidth() - 1, xPos + w / 2),
//                                Math.min(bitmap.getHeight() - 1, yPos + h / 2));
                float wRatio = width /INPUT_SIZE;
                float hRatio = height /INPUT_SIZE;
                final RectF rect =
                        new RectF(
                                Math.max(0, (xPos - w / 2)*wRatio),
                                Math.max(0, (yPos - h / 2)*hRatio),
                                Math.min(bitmap.getWidth() - 1, (xPos + w / 2)*wRatio),
                                Math.min(bitmap.getHeight() - 1, (yPos + h / 2)*hRatio));
                detections.add(new Recognition("" + offset, labels.get(detectedClass),
                        confidenceInClass, rect, detectedClass));
                Log.d("Class Detection: ", "class: " + labels.get(detectedClass) + " conf: " + confidenceInClass +
                        " Rect: " + rect + "DetectedClass: " + detectedClass);
            }
        }

        Log.d("YoloV5Classifier", "detect end");
        final ArrayList<Recognition> recognitions = nms(detections);
        ///////////////////////////////////////////////////////return recognitions;


        for(Recognition result : recognitions){
            final RectF location = result.getLocation();
            Log.d("Rectangle D", "left: " + location.left + " top: " + location.top + " right: " + location.right + " bottom: " + location.bottom);
            if (location != null && result.getConfidence() >= objThresh) {

                Log.d("rect info", "top: " + location.top + " left: " + location.left + " bottom: " + location.bottom + " right: " + location.right);

                Imgproc.rectangle(mat_image,new Point(location.left, location.top),new Point(location.right,location.bottom),new Scalar(0, 255, 0, 255),2);
                Imgproc.putText(mat_image,result.getTitle(),new Point(location.left,location.top),3,1,new Scalar(255, 0, 0, 255),2);
            }
        }
        return mat_image;
    }


    protected ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
                byteBuffer.putFloat((val & 0xFF) / 255.0f);
            }
        }
        return byteBuffer;
    }


    //non maximum suppression
    protected ArrayList<Recognition> nms(ArrayList<Recognition> list) {
        ArrayList<Recognition> nmsList = new ArrayList<>();

        for (int k = 0; k < labels.size(); k++) {
            //1.find max confidence per class
            PriorityQueue<Recognition> pq =
                    new PriorityQueue<>(
                            50,
                            new Comparator<Recognition>() {
                                @Override
                                public int compare(final Recognition lhs, final Recognition rhs) {
                                    // Intentionally reversed to put high confidence at the head of the queue.
                                    return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                                }
                            });

            for (int i = 0; i < list.size(); ++i) {
                if (list.get(i).getDetectedClass() == k) {
                    pq.add(list.get(i));
                }
            }

            //2.do non maximum suppression
            while (pq.size() > 0) {
                //insert detection with max confidence
                Recognition[] a = new Recognition[pq.size()];
                Recognition[] detections = pq.toArray(a);
                Recognition max = detections[0];
                nmsList.add(max);
                pq.clear();

                for (int j = 1; j < detections.length; j++) {
                    Recognition detection = detections[j];
                    RectF b = detection.getLocation();
                    if (box_iou(max.getLocation(), b) < mNmsThresh) {
                        pq.add(detection);
                    }
                }
            }
        }
        return nmsList;
    }

    protected float mNmsThresh = 0.6f;

    protected float box_iou(RectF a, RectF b) {
        return box_intersection(a, b) / box_union(a, b);
    }

    protected float box_intersection(RectF a, RectF b) {
        float w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top);
        if (w < 0 || h < 0) return 0;
        return w * h; //Area
    }

    protected float box_union(RectF a, RectF b) {
        float i = box_intersection(a, b);
        return  (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
    }

    protected float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = Math.max(l1, l2);
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = Math.min(r1, r2);
        return right - left;
    }


}
