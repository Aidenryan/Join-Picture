#include <iostream>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include<opencv2/xfeatures2d.hpp>
using namespace std;
typedef struct
{
    cv::Point2f left_top;
    cv::Point2f left_bottom;
    cv::Point2f right_top;
    cv::Point2f right_bottom;
}four_corners_t;

four_corners_t corners;

void merge_image(cv::Mat & fundhomography_matrix, vector<cv::Mat>& channels,cv::Mat& mergeimage,cv::Size image_size){

    cv::warpPerspective(channels[0],channels[0],fundhomography_matrix,image_size);
    cv::warpPerspective(channels[1],channels[1],fundhomography_matrix,image_size);
    cv::warpPerspective(channels[2],channels[2],fundhomography_matrix,image_size);
    ///合并色彩通道
    cv::merge(channels, mergeimage);

}
void Imagemosaic(cv::Mat & trans,cv::Mat & des){
    ///图像拼接
    for (int j=0;j!=des.rows;++j)
        for(int i=0;i!=des.cols;++i){
            if((trans.at<cv::Vec3b>(j,i)[0]==0&&trans.at<cv::Vec3b>(j,i)[1]==0&&trans.at<cv::Vec3b>(j,i)[2]==0))
            {
                trans.at<cv::Vec3b>(j,i)=des.at<cv::Vec3b>(j,i);
            }
        }
}


int main(int argc,char** argv)
{
    ///读取图像
    const string src_numeber="9";
    const string filepath="../../data/";
    const string src_name=filepath+src_numeber+"_1.jpg";
    const string des_name=filepath+src_numeber+"_2.jpg";
    const string mtcsopt_name=filepath+src_numeber+"_mtc_sopt.jpg";
    const string mtcini_name=filepath+src_numeber+"_mtc_ini.jpg";
    const string mtcfopt_name=filepath+src_numeber+"_mtc_fopt.jpg";
    const string ext_name=filepath+src_numeber+"_fea.jpg";
    const string merge_name=filepath+src_numeber+"_12.jpg";
    cv::Mat src=cv::imread(src_name);
    cv::Mat des=cv::imread(des_name);
    if(src.channels()<3){
        cv::cvtColor(src,src,cv::COLOR_GRAY2BGR);
    }
    if(des.channels()<3){
        cv::cvtColor(des,des,cv::COLOR_GRAY2BGR);
    }
    if(src.empty()||des.empty()){
        cout<<"image is empty!"<<endl;
        return 0;
    }
    ///提取灰度图像
    cv::Mat des_gray,src_gray;
    cv::cvtColor(src,src_gray,cv::COLOR_BGR2GRAY);
    cv::cvtColor(des,des_gray,cv::COLOR_BGR2GRAY);

    ///通道分离
    vector<cv::Mat> src_channels;
    cv::split(src,src_channels);

    vector<cv::Mat> des_channels;
    cv::split(des,des_channels);


    cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();


    vector<cv::KeyPoint> keypoints_src,keypoints_des;
    cv::Mat descriptor_src,descriptor_des;
    ///提取特征点
    sift->detect(src_gray,keypoints_src);
    sift->detect(des_gray,keypoints_des);
    ///计算描述子
    sift->detectAndCompute(src_gray,cv::Mat(),keypoints_src,descriptor_src);
    sift->detectAndCompute(des_gray,cv::Mat(),keypoints_des,descriptor_des);

    ///匹配
    //如果采用flannBased方法 那么 desp通过orb的到的类型不同需要先转换类型
    if (descriptor_src.type() != CV_32F || descriptor_des.type() != CV_32F)
    {
        descriptor_src.convertTo(descriptor_src, CV_32F);
        descriptor_des.convertTo(descriptor_des, CV_32F);
    }

    cv::Mat outing_ext;
    drawKeypoints(src_gray,keypoints_src,outing_ext,cv::Scalar::all(-1),cv::DrawMatchesFlags::DEFAULT);
    imshow("ORB keypoint",outing_ext);
    cv::imwrite(ext_name,outing_ext);
    ///筛选最佳匹配(1)
    std::vector<cv::DMatch> matches;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
    int k=2;
    matcher->match(descriptor_src, descriptor_des, matches);

    vector< vector<cv::DMatch> > nn_matches;
    std::vector<cv::DMatch> good_nnmatches ;
    matcher->knnMatch(descriptor_src, descriptor_des, nn_matches,2);

//////////画图
    cv::Mat img_match;
    cv::drawMatches(src_gray,keypoints_src,des_gray,keypoints_des,nn_matches,img_match);
    imshow("matches", img_match);
    cv::imwrite(mtcini_name,img_match);
///////////
    for (size_t i = 0; i < nn_matches.size(); i++) {
        cv::DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;

        if (dist1 < 0.75 * dist2) {
            good_nnmatches.push_back(first);
        }
    }
    cout<<" good match1: "<<good_nnmatches.size()<<endl;
    if(good_nnmatches.empty()){
        cout<<"no good nnmatch1!"<<endl;
        return 0;
    }
    else if(good_nnmatches.size()<10)
    {
        cout<<"no enough good nnmatch1!"<<endl;
        return 0;
    }



    ///筛选最佳匹配(2)
    auto min_max=minmax_element(good_nnmatches.begin(),good_nnmatches.end(),[](const cv::DMatch &m1,const cv::DMatch &m2){return m1.distance<m2.distance;});

    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;
    cout<<"min_dist: "<<min_dist<<endl;
    cout<<"max_dist: "<<max_dist<<endl;
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < good_nnmatches.size(); i++) {
        if(good_nnmatches[i].distance<MAX(2*min_dist,max_dist*0.9))
        {
            good_matches.push_back(good_nnmatches[i]);
        }
    }
    cout<<" good match2: "<<good_matches.size()<<endl;
    if(good_matches.empty()){
        cout<<"no good match2!"<<endl;
        return 0;
    }
    else if(good_matches.size()<10)
    {
        cout<<"no enough good match2!"<<endl;
        return 0;
    }

    ///此筛选可提高精确度，可视情况去掉
   /* good_nnmatches.clear();
    for (int i=0;i<good_matches.size();++i) {
        good_nnmatches.push_back(good_matches[i]);
    }*/


    cv::Mat img_goodmatch;
    cv::drawMatches(src_gray,keypoints_src,des_gray,keypoints_des,good_nnmatches,img_goodmatch);

    imshow("good matches", img_goodmatch);
    cv::imwrite(mtcfopt_name,img_goodmatch);
   // cv::waitKey(0);



    //-- 把匹配点转换为vector<Point2f>的形式
    vector<cv::Point2f> points1;
    vector<cv::Point2f> points2;

    ///计算透视矩阵
    for (int i = 0; i < (int) good_nnmatches.size(); i++) {
        points1.push_back(keypoints_src[good_nnmatches[i].queryIdx].pt);
        points2.push_back(keypoints_des[good_nnmatches[i].trainIdx].pt);
    }
    cv::Mat fundhomography_matrix;
    fundhomography_matrix = cv::findHomography(points1, points2, cv::RANSAC);
    cout << "fundhomography_matrix is " << endl << fundhomography_matrix << endl;

    if (fundhomography_matrix.type() != CV_64F )
    {
        fundhomography_matrix.convertTo(fundhomography_matrix, CV_64F);
    }

    ///2次筛选匹配点
    std::vector<cv::DMatch> good_bestmatches ;
    for (int i=0;i<points1.size();i++) {
        cv::Mat col = cv::Mat::ones(3, 1, CV_64F);
        col.at<double>(0) = points1[i].x;
        col.at<double>(1) = points1[i].y;

        col = fundhomography_matrix * col;
        col /= col.at<double>(2);
        double dist = sqrt(pow(col.at<double>(0) - points2[i].x, 2) +
                           pow(col.at<double>(1) - points2[i].y, 2));
        if (dist <= 5) {

            good_bestmatches.push_back(good_nnmatches[i]);
        }

    }
    cout<<" good best matches: "<<good_bestmatches.size()<<endl;
    if(good_bestmatches.empty()){
        cout<<"no best match!"<<endl;
        return 0;
    }
    else if(good_bestmatches.size()<6)
    {
        cout<<"no enough best match!"<<endl;
        return 0;
    }



    cv::drawMatches(src_gray,keypoints_src,des_gray,keypoints_des,good_bestmatches,img_goodmatch);
    imshow("good bestmatchs", img_goodmatch);
    cv::imwrite(mtcsopt_name,img_goodmatch);

   // cv::waitKey(0);
    ///二次计算透视矩阵
    points1.clear();
    points2.clear();
    for (int i = 0; i < (int) good_bestmatches.size(); i++) {
        points1.push_back(keypoints_src[good_bestmatches[i].queryIdx].pt);
        points2.push_back(keypoints_des[good_bestmatches[i].trainIdx].pt);
    }
    fundhomography_matrix = cv::findHomography(points1, points2, cv::RANSAC);
    cout << "best fundhomography_matrix is " << endl << fundhomography_matrix << endl;


    cv::Mat imagePesSrc;
    if(fundhomography_matrix.empty())
    {
        cout << "best fundhomography_matrix is  empty and try inverse !"<< endl;
    }
    else {
          ///图片透视变换与通道合并
          merge_image(fundhomography_matrix,src_channels,imagePesSrc,cv::Size(des.cols,des.rows));
          imshow("直接经过透视矩阵变换", imagePesSrc);
          ///图片拼接
          Imagemosaic(imagePesSrc,des);
    }


    cv::imshow("lookintopast", imagePesSrc);
    cv::imwrite(merge_name,imagePesSrc);

    cv::waitKey(0);



    return 0;
}

