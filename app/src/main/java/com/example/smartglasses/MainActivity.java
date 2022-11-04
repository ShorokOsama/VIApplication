package com.example.smartglasses;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import org.opencv.android.OpenCVLoader;

public class MainActivity extends AppCompatActivity {

    Button faceIdBtn, faceExpBtn, obstacleDetectionBtn;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Log.d("OpenCVModule", "OpenCV Loading Status: " + OpenCVLoader.initDebug());


        faceIdBtn = findViewById(R.id.face_id_btn);
        faceExpBtn = findViewById(R.id.face_expression_btn);
        obstacleDetectionBtn = findViewById(R.id.obstacle_detection_btn);


        faceIdBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intentID = new Intent(MainActivity.this, FaceIdActivity.class);
                startActivity(intentID);
            }
        });

        faceExpBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intentExp = new Intent(MainActivity.this, FaceExpressionActivity.class);
                startActivity(intentExp);
            }
        });

        obstacleDetectionBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intentDet = new Intent(MainActivity.this, ObstacleDetectionActivity.class);
                startActivity(intentDet);
            }
        });

    }
}