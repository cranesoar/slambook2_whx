#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
//#include "extra.h" // use this if in OpenCV2

using namespace std;
using namespace cv;

Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1); // 相机内参,TUM Freiburg2
Point2d principal_point(325.1, 249.7);  //相机光心, TUM dataset标定值　
const double focal_length = 521;      //相机焦距, TUM dataset标定值
const double SENSITIVITY = 0.1; //判断相机是否装反的阈值

void find_feature_matches(
  const Mat &img_right, const Mat &img_left,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

void pose_estimation_2d2d(
  std::vector<KeyPoint> keypoints_1,
  std::vector<KeyPoint> keypoints_2,
  std::vector<DMatch> matches,
  Mat &R, Mat &t);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

int main(int argc, char **argv) {
  if (argc != 3) {
    cout << "usage: pose_estimation_2d2d img_right img_left" << endl;
    return 1;
  }

  //-- 读取图像　
  Mat img_right = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat img_left = imread(argv[2], CV_LOAD_IMAGE_COLOR);
  assert(img_right.data && img_left.data && "Can not load images!");

  
  //   cvUndistort2(img_right, img_right, K);  //去畸变
  //   cvUndistort2(_img_left, _img_left, K);

  //匹配特征点
  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_right, img_left, keypoints_1, keypoints_2, matches);
  cout << "一共找到了" << matches.size() << "组匹配点" << endl;

  //-- 估计两张图像间运动
  Mat R, t;
  pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

  //判断左右相机相对位置　若img_right和img_left加载正确的双目对应相机图像，则运动矩阵ｔ的ｘ轴数值为正值，否则反之。
  if (t.at<double>(0, 0) < - SENSITIVITY)
  cout << "wrong  " << t.at<double>(0, 0) <<endl;
  else if(t.at<double>(0, 0) > SENSITIVITY)
  cout << "correct   "<< t.at<double>(0, 0) << endl;
  else
  cout << "NAN"<< t.at<double>(0, 0) << endl;

  waitKey(0);

  return 0;
}

void find_feature_matches(const Mat &img_right, const Mat &img_left,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
  //-- 初始化
  Mat descriptors_1, descriptors_2;
  // used in OpenCV3
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  // use this if you are in OpenCV2
  // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
  // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  //-- 第一步:检测 Oriented FAST 角点位置
  detector->detect(img_right, keypoints_1);
  detector->detect(img_left, keypoints_2);

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_right, keypoints_1, descriptors_1);
  descriptor->compute(img_left, keypoints_2, descriptors_2);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> match;
  //BFMatcher matcher ( NORM_HAMMING );
  matcher->match(descriptors_1, descriptors_2, match);

  //-- 第四步:匹配点对筛选
  double min_dist = 10000, max_dist = 0;

  //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (match[i].distance <= max(2 * min_dist, 30.0)) {
      matches.push_back(match[i]);
    }
  }

    //-- 第五步:绘制匹配结果
  Mat img_match;
  drawMatches(img_right, keypoints_1, img_left, keypoints_2, matches, img_match);
  imshow("all matches", img_match);
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
                          std::vector<KeyPoint> keypoints_2,
                          std::vector<DMatch> matches,
                          Mat &R, Mat &t) {

  //-- 把匹配点转换为vector<Point2f>的形式
  vector<Point2f> points1;
  vector<Point2f> points2;

  for (int i = 0; i < (int) matches.size(); i++) {
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }

  //-- 计算基础矩阵
  Mat fundamental_matrix;
  fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
  //cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

  //-- 计算本质矩阵
  Mat essential_matrix;
  essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
  //cout << "essential_matrix is " << endl << essential_matrix << endl;

  //-- 计算单应矩阵
  //-- 但是本例中场景不是平面，单应矩阵意义不大
  Mat homography_matrix;
  homography_matrix = findHomography(points1, points2, RANSAC, 3);
  //cout << "homography_matrix is " << endl << homography_matrix << endl;

  //-- 从本质矩阵中恢复旋转和平移信息.
  // 此函数仅在Opencv3中提供
  recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
  cout << "R is " << endl << R << endl;
  cout << "t is " << endl << t << endl;
   
}
