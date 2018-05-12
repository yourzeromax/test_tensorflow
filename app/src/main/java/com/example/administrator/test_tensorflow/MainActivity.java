package com.example.administrator.test_tensorflow;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Trace;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.MenuItem;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG ="123" ;

    static {
        System.loadLibrary("tensorflow_inference");
    }

    private CameraBridgeViewBase mOpenCvCameraView;
    private boolean mIsJavaCamera = true;
    private MenuItem mItemSwitchCamera = null;


    private static final String mode_file = "file:///android_asset/MnistTF_model.pb";
    private static final int NUM_CLASSES = 10;

    private int logit;   //输出数组中最大值的下标
    private float[] inputs_data = new float[784];
    private float[] outputs_data = new float[NUM_CLASSES];

    private String OUTPUT_NODE = "dense_3_2/Softmax:0";

    private TensorFlowInferenceInterface inferenceInterface;
    private String outputs_name = "dense_3_2/Softmax:0";

    private BaseLoaderCallback mLoaderCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            //super.onManagerConnected(status);
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    Log.i(TAG, "openCV loaded successfully ");
                    mOpenCvCameraView.enableView();
                    break;
                default: {
                        super.onManagerConnected(status);
                }break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        if(checkSelfPermission(Manifest.permission.CAMERA)!= PackageManager.PERMISSION_GRANTED){
            requestPermissions(new String[]{Manifest.permission.CAMERA},100);
        }
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutoriall_activity_java_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setMaxFrameSize(640,640);

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), mode_file);
        getPicturePixel();

        //   Trace.beginSection("feed");
        //输入节点名称 输入数据  数据大小
        //填充数据 1，784为神经网络输入层的矩阵大小
        inferenceInterface.feed("conv2d_1_input_2:0", inputs_data, 1, 28, 28, 1);
        inferenceInterface.run(new String[]{"dense_3_2/Softmax:0"});
        inferenceInterface.fetch(outputs_name, outputs_data);


//        Trace.beginSection("run");
//     //   Trace.endSection();
//
//        Trace.beginSection("fetch");
//
//        Trace.endSection();
        //取出数据
        //输出节点名称 输出数组
        // Trace.endSection();

        logit = 0;
        //找出预测的结果
        for (int i = 1; i < 10; i++) {
            if (outputs_data[i] > outputs_data[logit])
                logit = i;
        }


        //  inferenceInterface .feed();

    }

    @Override
    protected void onPause() {
        super.onPause();
        if(mOpenCvCameraView!=null){
            mOpenCvCameraView.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if(!OpenCVLoader.initDebug()){
            Log.d(TAG, "onResume: "+"init");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0,this,mLoaderCallBack);
        }else{
            Log.d(TAG, "onResume: "+"using it");
            mLoaderCallBack.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if(mOpenCvCameraView!=null){
            mOpenCvCameraView.disableView();
        }
    }

    private void getPicturePixel() {
        try {
            Resources res = getResources();
            Bitmap bitmap = BitmapFactory.decodeResource(res, R.mipmap.picture8);

            int width = bitmap.getWidth();
            int height = bitmap.getHeight();

            // 保存所有的像素的数组，图片宽×高
            int[] pixels = new int[width * height];

            bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

            for (int i = 0; i < pixels.length; i++) {
                inputs_data[i] = (float) pixels[i];
            }
        } catch (Exception e) {
            Log.d("tag", e.getMessage());
        }

    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat img_rgb = inputFrame.rgba();
        Mat img_t = new Mat();
        Mat img_gray = new Mat();
        Mat img_contours;

        Core.transpose(img_rgb,img_t);//转置函数，可以水平的图像变为垂直
        Imgproc.resize(img_t, img_rgb, img_rgb.size(), 0.0D, 0.0D, 0);
        Core.flip(img_rgb, img_rgb,1);  //flipCode>0将mRgbaF水平翻转（沿Y轴翻转）得到mRgba

        if(img_rgb != null) {
            Imgproc.cvtColor(img_rgb, img_gray, Imgproc.COLOR_RGB2GRAY);

            Imgproc.threshold(img_gray, img_gray, 140, 255, Imgproc.THRESH_BINARY_INV);

            //像素加强
            Mat ele1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
            Mat ele2 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(6, 6));
            Imgproc.erode(img_gray, img_gray, ele1);
            Imgproc.dilate(img_gray, img_gray, ele2);

            //找到外界矩形
            img_contours = img_gray.clone();
            List<MatOfPoint> contours = new ArrayList<>();
            Imgproc.findContours(img_contours, contours, new Mat(),
                    Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

            for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++) {
                double contourArea = Imgproc.contourArea(contours.get(contourIdx));
                Rect rect = Imgproc.boundingRect(contours.get(contourIdx));
                if (contourArea < 1500 || contourArea > 20000)
                    continue;

                Mat roi = new Mat(img_gray, rect);
                Imgproc.resize(roi, roi, new Size(28, 28));

                Bitmap bitmap2 = Bitmap.createBitmap(roi.width(), roi.height(), Bitmap.Config.RGB_565);
                Utils.matToBitmap(roi, bitmap2);
                int number = toNumber(bitmap2);
                if (number >= 0) {
                    //tl左上角顶点  br右下角定点
                    double x = rect.tl().x;
                    double y = rect.br().y;
                    Point p = new Point(x, y);
                    Imgproc.rectangle(img_rgb, rect.tl(), rect.br(), new Scalar(0, 0, 255));
                    Imgproc.putText(img_rgb, Integer.toString(number), p, Core.FONT_HERSHEY_DUPLEX,
                            6, new Scalar(0, 0, 255), 2);
                }
            }
            img_contours.release();
        }


        img_gray.release();
        img_t.release();
        img_t.release();
        return  img_rgb;
    }

    //28 X 28   1
    //给一个 bitmap   识别成数字  以int形式返回
    int toNumber(Bitmap bitmap_roi){
        int width = bitmap_roi.getWidth();
        int height = bitmap_roi.getHeight();
        int[] pixels = new int[width * height];

        Log.d("tag", width+"  "+height);

        try {
            bitmap_roi.getPixels(pixels, 0, width, 0, 0, width, height);
            for (int i = 0; i < pixels.length; i++) {
                inputs_data[i] = (float)pixels[i];
            }
        }catch (Exception e){
            Log.d("tag", e.getMessage());
        }

        Log.d("Tag", "width: "+width+"   height:"+height);

        Trace.beginSection("feed");
        inferenceInterface.feed("conv2d_1_input_2:0", inputs_data, 1,28,28,1);
        Trace.endSection();

        Trace.beginSection("run");
        inferenceInterface.run(new String[]{OUTPUT_NODE});
        Trace.endSection();

        Trace.beginSection("fetch");
        inferenceInterface.fetch(OUTPUT_NODE, outputs_data);
        Trace.endSection();

        int logit = 0;
        for(int i=1;i<10;i++)
        {
            if(outputs_data[i]>outputs_data[logit])
                logit=i;
        }

        if(outputs_data[logit]>0)
            return logit;
        return -1;

    }
}