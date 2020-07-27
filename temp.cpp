#include <iostream>
#include <fstream> 
#include <string>
#include <iomanip> 
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
using namespace cv;
using namespace std;
using namespace detail;
 
int main(int argc, char** argv)
{
   vector<Mat> imgs;    //输入图像序列
   Mat img;

   String path = argv[1]; //读取图像
   vector<String> filesname;
   glob(path,filesname,false);
   for( auto n : filesname)
   {
      img = imread( n );
      imgs.push_back( img );
   }
   int num_images = imgs.size();    //图像数量
   cout<<"待拼接图像为："<<num_images<<"张"<<endl;
   Ptr<FeaturesFinder> finder;    //定义特征寻找器
   
   finder = new OrbFeaturesFinder();    //寻找ORB特征
   vector<ImageFeatures> features(num_images);    //表示图像特征
   for (int i =0 ;i<num_images;i++)
      (*finder)(imgs[i], features[i]);    //特征检测
   cout<<"特征提取完毕"<<endl;

   vector<MatchesInfo> pairwise_matches;    //特征匹配信息
   BestOf2NearestMatcher matcher(false, 0.3f, 6, 6);    //特征匹配器，采用最近邻和次近邻法的方法
   matcher(features, pairwise_matches);    //进行特征匹配
   cout<<"特征匹配完毕"<<endl;

   HomographyBasedEstimator estimator;    //基于单应性矩阵的参数估计
   vector<CameraParams> cameras;    //表示相机参数（内参和外参）
   estimator(features, pairwise_matches, cameras);    //估计相机参数
 
   for (size_t i = 0; i < cameras.size(); ++i)    //转换相机旋转参数的数据类型
   {
      Mat R;
      cameras[i].R.convertTo(R, CV_32F);
      cameras[i].R = R;  
   }
 
   Ptr<detail::BundleAdjusterBase> adjuster;    //采用光束平差法，精确相机参数
   adjuster = new detail::BundleAdjusterReproj();    //最小化重映射误差
   
    
   adjuster->setConfThresh(1);    //设置匹配置信度，该值设为1
   (*adjuster)(features, pairwise_matches, cameras);    //精确评估相机参数
 
   vector<Mat> rmats;
   for (size_t i = 0; i < cameras.size(); ++i)    //复制相机的旋转参数
      rmats.push_back(cameras[i].R.clone());
   waveCorrect(rmats, WAVE_CORRECT_HORIZ);    //进行波形校正

   for (size_t i = 0; i < cameras.size(); ++i)    //相机参数赋值
      cameras[i].R = rmats[i];
   rmats.clear();   
 
   vector<Point> corners(num_images);    //投影变换后图像的左上角坐标
   vector<UMat> masks_warped(num_images);    //投影变换后的图像掩码
   vector<UMat> images_warped(num_images);    //投影变换后的图像
   vector<Size> sizes(num_images);    //投影变换后的图像尺寸
   vector<Mat> masks(num_images);    //源图的掩码
 
   for (int i = 0; i < num_images; ++i)    //初始化源图的掩码
   {
      masks[i].create(imgs[i].size(), CV_8U);    //定义尺寸大小
      masks[i].setTo(Scalar::all(255));    //全部赋值为255，表示源图的所有区域都使用
   }
 
   Ptr<WarperCreator> warper_creator;    //定义图像映射变换创造器
   warper_creator = new cv::PlaneWarper();    //平面投影

 
   //定义图像映射变换器，设置映射的尺度为相机的焦距，所有相机的焦距都相同
   Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(cameras[0].focal));
   for (int i = 0; i < num_images; ++i)
    {
      Mat_<float> K;
      cameras[i].K().convertTo(K, CV_32F);    //转换相机内参数的数据类型
      //对当前图像镜像投影变换，得到变换后的图像以及该图像的左上角坐标
      corners[i] = warper->warp(imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
      sizes[i] = images_warped[i].size();    //得到尺寸
      //得到变换后的图像掩码
      warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
   }
	
   imgs.clear();    //清变量
   masks.clear();
 
   cout<<"图像投影完毕"<<endl;

   //创建曝光补偿器，应用增益补偿方法
   Ptr<ExposureCompensator> compensator =       
                     ExposureCompensator::createDefault(ExposureCompensator::GAIN);
   compensator->feed(corners, images_warped, masks_warped);    //得到曝光补偿器
   for(int i=0;i<num_images;++i)    //应用曝光补偿器，对图像进行曝光补偿
   {
      compensator->apply(i, corners[i], images_warped[i], masks_warped[i]);
   }
   

   //在后面，我们还需要用到映射变换图的掩码masks_warped，因此这里为该变量添加一个masks_seam
   vector<UMat> masks_seam(num_images); 
   for(int i = 0; i<num_images;i++)
      masks_warped[i].copyTo(masks_seam[i]);
 
   Ptr<SeamFinder> seam_finder;    //定义接缝线寻找器
   
   //图割法
   //seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR);
   seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR_GRAD);
    
   vector<UMat> images_warped_f(num_images);
   for (int i = 0; i < num_images; ++i)    //图像数据类型转换
      images_warped[i].convertTo(images_warped_f[i], CV_32F);
 
   images_warped.clear();    //清内存
 
   //得到接缝线的掩码图像masks_seam
   seam_finder->find(images_warped_f, corners, masks_seam); 
	cout<<"拼缝优化完毕"<<endl;

   vector<Mat> images_warped_s(num_images);
   Ptr<Blender> blender;    //定义图像融合器
 
   blender = Blender::createDefault(Blender::MULTI_BAND, false);    //多频段融合
   MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
   mb->setNumBands(8);   //设置频段数，即金字塔层数
 

   blender->prepare(corners, sizes);    //生成全景图像区域
   cout<<"生成全景图像区域"<<endl;

   //在融合的时候，最重要的是在接缝线两侧进行处理，而上一步在寻找接缝线后得到的掩码的边界就是接缝线处，
   //因此需要在接缝线使用膨胀算法在两侧开辟一块区域用于融合处理。
   vector<Mat> dilate_img(num_images);
   vector<Mat> masks_seam_new(num_images);
   Mat tem;
   Mat element = getStructuringElement(MORPH_RECT, Size(20, 20));    //定义结构元素
   for(int k=0;k<num_images;k++)
   {
      images_warped_f[k].convertTo(images_warped_s[k], CV_16S);    //改变数据类型
      dilate(masks_seam[k], masks_seam_new[k], element);    //膨胀运算
      //映射变换图的掩码和膨胀后的掩码相“与”，从而使扩展的区域仅仅限于接缝线两侧，其他边界处不受影响
      masks_warped[k].copyTo(tem);
      masks_seam_new[k] = masks_seam_new[k] & tem;
      blender->feed(images_warped_s[k], masks_seam_new[k], corners[k]);    //初始化数据
      cout<<"处理完成"<<k<<"图片"<<endl;
   }
   masks_seam.clear();    //清内存
   images_warped_s.clear();
   masks_warped.clear();
   images_warped_f.clear();
   

   Mat result, result_mask;
   //完成融合操作，得到全景图像result和它的掩码result_mask
   blender->blend(result, result_mask);
 
   imwrite("pano.jpg", result);    //存储全景图像

   Mat plus_re;

   //在全景图四周各添加30像素宽的黑色边框，以确保能够找到全景图的完整轮廓
   cv::copyMakeBorder(result,plus_re,30,30,30,30,BORDER_CONSTANT,(0,0,0));
   imwrite("plus_re.jpg",plus_re);

   Mat gray_re;
   convertScaleAbs(plus_re,plus_re, 1, 0);//改变图像深度，cv::tcolor只允许0,2,5类型图像输入
   //cout<<"类型： "<<plus_re.depth()<<endl;
   cv::cvtColor(plus_re,gray_re,COLOR_BGR2GRAY);

   Mat binary_re;
   cv::threshold(gray_re,binary_re,0,255,THRESH_BINARY);
   imwrite("binary_re.jpg", binary_re);

   vector<vector<Point> > contours;
   vector<Vec4i> hierarchy;

   //找到最大轮廓的边界框
   findContours( binary_re, contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE );
   double maxarea = 0;
	int maxAreaIdx = 0;
	for (int index = contours.size() - 1; index >= 0; index--)
	{
		double tmparea = fabs(contourArea(contours[index]));
		if (tmparea>maxarea)
		{
			maxarea = tmparea;
			maxAreaIdx = index;//记录最大轮廓的索引号
		}
	}
   vector<Point> contmax = contours[maxAreaIdx];
   cout<<"获取最大轮廓完毕"<<endl;

   Mat mask=Mat::zeros(binary_re.rows,binary_re.cols,CV_8U);   
   Rect ret = boundingRect(contmax);

   rectangle(mask, cvPoint(ret.x,ret.y), cvPoint(ret.x+ret.width,ret.y+ret.height), 255, -1);
   imwrite("maks.jpg",mask);
   
   //最终想要的全景图内部最大矩形区域
   Mat minRect, sub;
   mask.copyTo(minRect);mask.copyTo(sub);
   Mat structElement2 = getStructuringElement(MORPH_RECT, Size(3,3), Point(-1,-1));
   while(countNonZero(sub)>0)
   {
      erode(minRect, minRect,structElement2);
      subtract(minRect, binary_re, sub);
   }
  
   imwrite("min.jpg",minRect);
   vector<vector<Point> > contours1;
   vector<Vec4i> hierarchy1;
   findContours( minRect, contours1, hierarchy1, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE );
   double maxarea1 = 0;
	int maxAreaIdx1 = 0;
	for (int index = contours1.size() - 1; index >= 0; index--)
	{
		double tmparea = fabs(contourArea(contours1[index]));
		if (tmparea>maxarea)
		{
			maxarea = tmparea;
			maxAreaIdx = index;//记录最大轮廓的索引号
		}
	}
   vector<Point> contmax1 = contours1[maxAreaIdx1];
   Rect ret1 = boundingRect(contmax1);
   cout<<"找到最大完整区域"<<endl;
   //cout<<ret1.x<<" "<<ret1.y<<" "<<ret1.x+ret1.width<<" "<<ret1.y+ret1.height<<endl;
   Rect rr(ret1.x,ret1.y, ret1.width, ret1.height);//取区域赋值，起点，如何在是宽和高
   //Rect rr(ret1.y,ret1.x, ret1.y+ret1.height,ret1.x+ret1.width);
   Mat last_re = result(rr);
   cout<<"最终结果完成"<<endl;
   imwrite("last_re.jpg", last_re);
   return 0;
}