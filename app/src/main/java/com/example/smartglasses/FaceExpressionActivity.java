package com.example.smartglasses;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.os.Bundle;
import android.util.Log;

import com.example.smartglasses.ml.ExpressionModel;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

public class FaceExpressionActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    JavaCameraView javaCameraView;
    File cascFile;

    int imageSize = 48;

    CascadeClassifier faceDetector;
    private Mat mRgba, mGrey;

    String[] labels = {"Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_face_expression);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, 1);

        javaCameraView = (JavaCameraView) findViewById(R.id.javaCamView);

        if(!OpenCVLoader.initDebug()){
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, baseCallback);
        }else{
            try {
                baseCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        javaCameraView.setCvCameraViewListener(this);


    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat();
        mGrey = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGrey.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGrey = inputFrame.gray();

        //Face Detection
        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(mRgba, faceDetections);

        for(Rect rect: faceDetections.toArray()){
            Rect roi = new Rect(rect.x, rect.y, rect.width, rect.height);
            Mat cropped = new Mat(mRgba, roi);
            Bitmap face = convertMatToBitMap(cropped);

            String faceExp = getFaceExp(face);

            Imgproc.rectangle(mRgba, new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(255,0,0), 3);
            Imgproc.putText(mRgba, faceExp, new Point(rect.x, rect.y), Core.FONT_HERSHEY_SIMPLEX,
                    1, new Scalar(255,0,0),3);
        }

        return mRgba;
    }

    private BaseLoaderCallback baseCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) throws IOException {
            switch (status){
                case LoaderCallbackInterface.SUCCESS:{
                    InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt2);
                    File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                    cascFile = new File(cascadeDir, "haarcascade_frontalface_alt2.xml");

                    FileOutputStream fos = new FileOutputStream(cascFile);

                    byte[] buffer = new byte[4096];
                    int bytesRead;

                    while((bytesRead = is.read(buffer)) != -1){
                        fos.write(buffer, 0, bytesRead);
                    }
                    is.close();
                    fos.close();

                    faceDetector = new CascadeClassifier(cascFile.getAbsolutePath());

                    if(faceDetector.empty()){
                        faceDetector = null;
                    }else{
                        cascadeDir.delete();
                    }
                    javaCameraView.enableView();

                    break;

                }
                default:{
                    super.onManagerConnected(status);
                }

            }
        }
    };

    public String getFaceExp(Bitmap img){
        String exp = "NULL";
        Bitmap greyFace = toGrayScale(img);

        try {
            ExpressionModel model = ExpressionModel.newInstance(this);

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 48, 48, 3}, DataType.FLOAT32);

            Bitmap input=Bitmap.createScaledBitmap(greyFace,48,48,true);
            TensorImage image=new TensorImage(DataType.FLOAT32);
            image.load(input);
            ByteBuffer byteBuffer = image.getBuffer();
            Log.d("shape", "byte buffer shape: " + byteBuffer.toString());
            Log.d("shape", "input feature shape: " + inputFeature0.getBuffer().toString());

//            inputFeature0.loadBuffer(TensorImage.fromBitmap(input).getBuffer());

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ExpressionModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            exp = labels[getMax(outputFeature0.getFloatArray())]+"";

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }

        return exp;
    }

    int getMax(float[] arr){
        int max = 0;
        for(int i = 0; i < arr.length; i++){
            if(arr[i] > arr[max]) max = i;
        }
        return max;
    }

    public static Bitmap toGrayScale(Bitmap bmpOriginal) {

        int width, height;
        height = bmpOriginal.getHeight();
        width = bmpOriginal.getWidth();

        Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmpGrayscale);
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
        paint.setColorFilter(f);
        c.drawBitmap(bmpOriginal, 0, 0, paint);
        bmpOriginal.recycle();
        return bmpGrayscale;
    }

    private static Bitmap convertMatToBitMap(Mat input){
        Bitmap bmp = null;
        Mat rgb = new Mat();
        Imgproc.cvtColor(input, rgb, Imgproc.COLOR_BGR2RGB);

        try {
            bmp = Bitmap.createBitmap(rgb.cols(), rgb.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(rgb, bmp);
        }
        catch (CvException e){
            Log.d("Exception",e.getMessage());
        }
        return bmp;
    }
}