package com.example.administrator.test_tensorflow;

import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Trace;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("tensorflow_inference");
    }

    private static final String mode_file = "file:///android_asset/MnistTF_model.pb";
    private static final int NUM_CLASSES = 10;

    private int logit;   //输出数组中最大值的下标
    private float[] inputs_data = new float[784];
    private float[] outputs_data = new float[NUM_CLASSES];
    private TensorFlowInferenceInterface inferenceInterface;
    private  String outputs_name = "dense_3_2/Softmax:0";



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        TextView text = (TextView) findViewById(R.id.textView);
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), mode_file);
        getPicturePixel();

        Trace.beginSection("feed");
        //输入节点名称 输入数据  数据大小
        //填充数据 1，784为神经网络输入层的矩阵大小
        inferenceInterface.feed("conv2d_1_input_2:0", inputs_data,1,28,28,1);
        inferenceInterface.run(new String[]{"dense_3_2/Softmax:0"});
        inferenceInterface.fetch(outputs_name,outputs_data);
        Trace.endSection();

        Trace.beginSection("run");
        Trace.endSection();

        Trace.beginSection("fetch");
        //取出数据
        //输出节点名称 输出数组
        Trace.endSection();

        logit = 0;
        //找出预测的结果
        for(int i=1;i<NUM_CLASSES;i++)
        {
            if(outputs_data[i]>outputs_data[logit])
                logit=i;
        }
        text.setText("The number is "+ logit);


        //  inferenceInterface .feed();

    }

    private void getPicturePixel() {
        try{
            Resources res = getResources();
            Bitmap bitmap = BitmapFactory.decodeResource(res,R.mipmap.picture8);
            ImageView img= (ImageView) findViewById(R.id.img);
            img.setImageBitmap(bitmap);

            int width = bitmap.getWidth();
            int height = bitmap.getHeight();

            // 保存所有的像素的数组，图片宽×高
            int[] pixels = new int[width * height];

            bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

            for (int i = 0; i < pixels.length; i++) {
                inputs_data[i] = (float)pixels[i];
            }
        }catch (Exception e){
            Log.d("tag", e.getMessage());
        }

    }

}
