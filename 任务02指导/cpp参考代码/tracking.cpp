#pragma once
/**
 ********************************************************************************************************
 *                                               示例代码
 *                                             EXAMPLE  CODE
 *
 *                      (c) Copyright 2024; SaiShu.Lcc.; Leo; https://bjsstech.com
 *                                   版权所属[SASU-北京赛曙科技有限公司]
 *
 *            The code is for internal use only, not for commercial transactions(开源学习,请勿商用).
 *            The code ADAPTS the corresponding hardware circuit board(代码适配百度Edgeboard-智能汽车赛事版),
 *            The specific details consult the professional(欢迎联系我们,代码持续更正，敬请关注相关开源渠道).
 *********************************************************************************************************
 * @file tracking.cpp
 * @author your name (you@domain.com)
 * @brief 赛道线识别：提取赛道左右边缘数据（包括岔路信息等）
 * @version 0.1
 * @date 2022-02-18
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <fstream>
#include <iostream>
#include <cmath>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "../../include/common.hpp"


using namespace cv;
using namespace std;

class Tracking
{
public:
#define pixel(row, col) imagePath.at<uchar>(row, col)
    vector<POINT> pointsEdgeLeft;       // 赛道左边缘点集（row，column）
    vector<POINT> pointsEdgeRight;      // 赛道右边缘点集
    vector<POINT> widthBlock;           // 色块宽度=终-起（每行）[0,319]
    vector<POINT> inlines;              // 记录赛道内的异常点
    vector<POINT> corners;              // 记录赛道边缘的异常点
    double stdevLeft;                   // 边缘斜率方差（左）
    double stdevRight;                  // 边缘斜率方差（右）
    double curvature_left;              // 边缘曲率（左）
    double curvature_right;             // 边缘曲率（右）
    POINT garageEnable = POINT(0, 0);   // 车库识别标志：（x=1/0，y=row)
    uint16_t rowCutUp = 10;             // 图像顶部切行
    uint16_t rowCutBottom = 10;         // 图像底部切行
    int maxblock_global = 0;            //一帧最宽色块
    int maxblockrow_global = 0;
    Rect box_roi;                     //inlines[0]的box
    POINT BottomPoint;
    int block_num =0;
    int blocks_num = 0;
    int block_one = 0;
    vector<uint16_t> pointsBlockFar;
    int FindBottomPoint(int right) {
	int index = 0;
	int col = 0;
	if (right) {
		col = pointsEdgeRight[0].y;
		for (int i = 1; i < pointsEdgeRight.size()*0.75; i++) {
			if (pointsEdgeRight[i].y < col) {
				col = pointsEdgeRight[i].y;
				index = i;
			}
		}
		POINT tmp(pointsEdgeRight[index].x, pointsEdgeRight[index].y);
		BottomPoint = tmp;
	}
	else {
		col = pointsEdgeLeft[0].y;
		for (int i = 1; i < pointsEdgeLeft.size(); i++) {
			if (pointsEdgeLeft[i].y > col) {
				col = pointsEdgeLeft[i].y;
				index = i;
			}
		}
		POINT tmp(pointsEdgeRight[index].x, pointsEdgeRight[index].y);
		BottomPoint = tmp;
	}
	return index;
}
    /**
     * @brief 赛道线识别
     *
     * @param isResearch 是否重复搜索
     * @param rowStart 边缘搜索起始行
     * return 赛道：pointsEdgeLeft，pointsEdgeRight,widthBlock，validRows（有效行数），边缘斜率/方差
     *        车库：garageEnable
     *        岔路：spurroadEnable，spurroad
     * -------注意---------limitWidthBlock------------取值------------
     */
    void trackRecognition(Mat &imageBinary, Scene scene,int parkstep, int parkside) {
      vector<POINT> mblock;
      bool flagStartBlock = true; // 搜索到色块起始行的标志（行）
      bool spurroadEnable = false;
      bool cornerEnable = false;
      int gap = -1;
      int rowStart = ROWSIMAGE - rowCutBottom;

      init_param();
      imagePath = imageBinary.clone();
      FilterbyDE(imagePath);
      switch (scene) {// corner的gap取值
      case Scene::NormalScene: 
          gap = 30;
          break;
      case Scene::CateringScene:
          gap = 20;
          break;
      case Scene::ObstacleScene:
          gap = 20;
          break;     
      default:
          gap = -1;
      }
      for (int row = rowStart; row > rowCutUp; row--) // 提取row范围
      {
        max_width = 0;
        indexWidestBlock = 0;
        block.clear();
        mblock.clear();
        FindBlockinRow(row);
        FindMaxBlockinRow();
        if (flagStartBlock) // 起始行做特殊处理
        {
          if (block.size() == 0 || max_width < COLSIMAGE * 0.25)
            continue;
          if (row < ROWSIMAGE / 2) //首行行数限制
            return;
          float white_ratio = WhiteRatioinRow(block);
          //提取赛道边缘
          if (white_ratio > 0.5 && scene != Scene::ParkingScene) { //比率满足时，障碍物在赛道内
            flagStartBlock = false;
            PushBlock(row, block[0].x, block[block.size() - 1].y);
          } else { //障碍物在赛道外
            maxblock_global = max_width;
            maxblockrow_global = row;
            flagStartBlock = false;
            PushBlock(row, block[indexWidestBlock].x,block[indexWidestBlock].y);
          }
          spurroadEnable = false;
        } 
        else // 其它行色块坐标处理
        {
          if (block.size() == 0)
            return;
          if (maxblock_global < max_width) {
            maxblock_global = max_width;
            maxblockrow_global = row;
          }
#if 1
          //-------------------------------------------------<车库标识识别>-------------------------------------------------------------
        //   LOGLN(block.size())
          if (block.size() > 5 && !garageEnable.x) {
            int widthThis = 0;        // 色块的宽度
            int widthVer = 0;         // 当前行色块的平均值
            vector<int> widthGarage;  // 当前行色块宽度集合
            vector<int> centerGarage; // 当前行色块质心集合
            vector<int> indexGarage;  // 当前行有效色块的序号

            for (int i = 0; i < block.size(); i++) //去噪
            {
              widthThis = block[i].y - block[i].x;        // 色块的宽度
              int center = (block[i].x + block[i].y) / 2; // 色块的质心
              if (widthThis > 5 &&
                  widthThis < 50) // 过滤无效色块区域：噪点，只保存5-50的色块
              {
                centerGarage.push_back(center);
                widthGarage.push_back(widthThis);
              }
            }

            int widthMiddle = getMiddleValue(widthGarage); // 斑马线色块宽度中值

            for (int i = 0; i < widthGarage.size(); i++) //滤波
            {
              if (abs(widthGarage[i] - widthMiddle) < widthMiddle / 3) {
                indexGarage.push_back(i);
              }
            }
            // LOGLN(indexGarage.size() );
            if (indexGarage.size() >= 4) // 验证有效斑马线色块个数
            {
              vector<int> distance;
              for (int i = 1; i < indexGarage.size(); i++) // 质心间距的方差校验
              {
                distance.push_back(widthGarage[indexGarage[i]] -
                                   widthGarage[indexGarage[i - 1]]);
              }
              double var = sigma(distance);
            //   LOGLN(var);
              if (var < 5.0) // 经验参数，检验均匀分布
              {
                garageEnable.x = 1;                      // 车库标志使能
                garageEnable.y = pointsEdgeRight.size(); // 斑马线行序号
                cornerEnable = true;
                spurroadEnable = true;
              }
            }
          }
          //------------------------------------------------------------------------------------------------------------------------
          int last_l = pointsEdgeLeft[pointsEdgeLeft.size() - 1].y;
          int last_r = pointsEdgeRight[pointsEdgeRight.size() - 1].y;
          // 上下行色块的连通性判断
          // if (scene == Scene::CateringScene) {
          //   int start_l = pointsEdgeLeft[0].y;
          //   int start_r = pointsEdgeRight[0].y;
          //   //LOGLN(start_l << start_r);
          //   for (int i = 0; i < block.size(); i++) {
          //     if (block[i].y > start_l && block[i].x < start_r)
          //       mblock.push_back(block[i]);
          //   }
          // } 
          for (int i = 0; i < block.size(); i++) {
            if (block[i].y > last_l && block[i].x < last_r)
              mblock.push_back(block[i]);
          }
          

          if (mblock.size() == 0) { // 如果没有发现联通色块，则图像搜索完成，结束任务
            return;
          } else if (mblock.size() == 1) { // 只存在单个色块，正常情况，提取边缘信息
            if (mblock[0].y - mblock[0].x < COLSIMAGE / 10) //排除异常短色块
              continue;
            PushBlock(row, mblock[0].x, mblock[0].y);
            // 边缘斜率计算
            slopeCal(pointsEdgeLeft, pointsEdgeLeft.size() - 1);
            slopeCal(pointsEdgeRight, pointsEdgeRight.size() - 1);

            FindCorner(gap);
            cornerEnable = false;
            spurroadEnable = false;
          } else if (mblock.size() >1) { // 存在多个色块，则需要择优处理：选取与上一行最近的色块
            //方案一：只使用最大的色块
            //int max_index = 0;
            //for (int i = 1; i < mblock.size(); i++) {
            //    if (mblock[i].y - mblock[i].x > mblock[max_index].y - mblock[max_index].x)
            //        max_index = i;
            //}
            //PushBlock(row, mblock[max_index].x, mblock[max_index].y);

            //方案二： 计算比率,满足则合并，否则用max
            if(scene == Scene::ParkingScene && parkstep == 3){
                if(parkside == 1)
                    PushBlock(row, mblock[mblock.size()-1].x, mblock[mblock.size()-1].y);
                else
                    PushBlock(row, mblock[0].x, mblock[0].y);    
            }else if(WhiteRatioinRow(mblock) > 0.8 || garageEnable.x)
                PushBlock(row, mblock[0].x, mblock[mblock.size()-1].y);
            else {
                int max_index = 0;
                for (int i = 1; i < mblock.size(); i++) {
                    if (mblock[i].y - mblock[i].x > mblock[max_index].y - mblock[max_index].x)
                        max_index = i;
                }
                PushBlock(row, mblock[max_index].x, mblock[max_index].y);
            }
            slopeCal(pointsEdgeLeft, pointsEdgeLeft.size() - 1);
            slopeCal(pointsEdgeRight, pointsEdgeRight.size() - 1);
            // if (!cornerEnable) {
            //     FindCorner(gap);
            //     cornerEnable = true;
            // }
            //-------------------------------<岔路信息提取>----------------------------------------
            if (!spurroadEnable) {
              for (int i = 1; i < mblock.size(); i++) {
                inlines.push_back(POINT(row, mblock[i].x));
              }
              spurroadEnable = true;
            }
            //------------------------------------------------------------------------------------
          }
#endif
          stdevLeft = stdevEdgeCal(pointsEdgeLeft, ROWSIMAGE); // 计算边缘方差
          stdevRight = stdevEdgeCal(pointsEdgeRight, ROWSIMAGE);
        }
        GetBlocksNum();
      }
      if(garageEnable.x){
        inlines.clear();
        corners.clear();
      }
    }
    //计算在赛道中的白色占比（当赛道中出现深色障碍物）
    float WhiteRatioinRow(vector<POINT>blocks) {
      int white_width = 0;
      for (int i = 0; i < blocks.size(); i++)
          white_width += blocks[i].y - blocks[i].x;
      int sum_width = blocks[blocks.size() - 1].y - blocks[0].x;
      return (float)white_width / sum_width;
    }
    void PushBlock(int row,int start,int end){
      POINT pointTmp(row, start);
      pointsEdgeLeft.push_back(pointTmp);
      pointTmp.y = end;
      pointsEdgeRight.push_back(pointTmp);
      widthBlock.emplace_back(row,end-start);
    } 
    void init_param(){
      pointsEdgeLeft.clear();              // 初始化边缘结果
      pointsEdgeRight.clear();             // 初始化边缘结果
      widthBlock.clear();                  // 初始化色块数据
      corners.clear();                    // 岔路信息
      inlines.clear();
      maxblock_global = 0;
      maxblockrow_global = 0;
      garageEnable = POINT(0, 0);          // 车库识别标志初始化

      block_num =0;
      blocks_num = 0;
      block_one  = 0;
      pointsBlockFar.clear();

      //切行
      if (rowCutUp > ROWSIMAGE / 4)
      rowCutUp = ROWSIMAGE / 4;
      if (rowCutBottom > ROWSIMAGE / 4)
      rowCutBottom = ROWSIMAGE / 4;
    }
    void FindMaxBlockinRow(){
      for (int i = 0; i < block.size(); i++){
          int tmp_width = block[i].y - block[i].x;
          if (tmp_width > max_width){
              max_width = tmp_width;
              indexWidestBlock = i;
          }
      }
    }
    //判断最后两个点斜率，如果相反且两点的距离满足条件则提取
    void FindCorner(int width) {
      if (pointsEdgeLeft.size() < 5)return;
      int last_y = pointsEdgeLeft[pointsEdgeLeft.size() - 1].y;
      int current_y = pointsEdgeLeft[pointsEdgeLeft.size() - 2].y;
      float tmp1_slope = pointsEdgeLeft[pointsEdgeLeft.size() - 1].slope;
      float tmp2_slope = pointsEdgeLeft[pointsEdgeLeft.size() - 2].slope;
      if (tmp1_slope * tmp2_slope < 0 && abs(last_y-current_y) > width) {
          corners.push_back(pointsEdgeLeft[pointsEdgeLeft.size() - 1]);
      }
      last_y = pointsEdgeRight[pointsEdgeRight.size() - 1].y;
      current_y = pointsEdgeRight[pointsEdgeRight.size() - 2].y;
      tmp1_slope = pointsEdgeRight[pointsEdgeRight.size() - 1].slope;
      tmp2_slope = pointsEdgeRight[pointsEdgeRight.size() - 2].slope;
      if (tmp1_slope * tmp2_slope < 0 && abs(last_y - current_y) > width) {
          POINT tmp(pointsEdgeRight[pointsEdgeRight.size() - 1].x, pointsEdgeRight[pointsEdgeRight.size() - 1].y);
          corners.push_back(pointsEdgeRight[pointsEdgeRight.size() - 1]);
      }
    }
    void FindBlockinRow(int row) {
      int black_limit = 5;
      int white_limit = 7;
      int start = 0;
      int tmp_end = 0;
      bool flag = false;
      for (int col = 1; col < COLSIMAGE; col++) {
          if (!flag) {
              if (pixel(row, col) > 127 && pixel(row, col - 1) <= 127) {
                  start = col;
              }
              else if (pixel(row, col) <= 127 && pixel(row, col - 1) > 127) {
                  if (col - start > white_limit) {
                      tmp_end = col;
                      flag = true;
                  }
              }
          }
          else {
              if (pixel(row, col) > 127 && pixel(row, col - 1) <= 127) {
                  if (col - tmp_end >= black_limit) {//滤波
                      POINT tmp(start, tmp_end);
                      block.push_back(tmp);
                      start = col;
                      flag = false;
                  }
              }
              else if (pixel(row, col) <= 127 && pixel(row, col - 1) > 127) {
                  tmp_end = col;
              }
          }
      }
      if (pixel(row, COLSIMAGE - 1) > 127 && COLSIMAGE - 1 - start > white_limit) {
              POINT tmp(start, COLSIMAGE - 1);
              block.push_back(tmp);
          
      }
      else if(pixel(row, COLSIMAGE - 1) < 127 && tmp_end - start > white_limit){
              POINT tmp(start, tmp_end);
              block.push_back(tmp);        
      }
    }

    /**
     * @brief 显示赛道线识别结果
     *
     * @param trackImage 需要叠加显示的图像
     */
    void drawImage(Mat& trackImage)
    {
        for (int i = 0; i < pointsEdgeLeft.size(); i++)
        {
            circle(trackImage, Point(pointsEdgeLeft[i].y, pointsEdgeLeft[i].x), 1,
                Scalar(0, 255, 0), -1); // 绿色点
        }
        for (int i = 0; i < pointsEdgeRight.size(); i++)
        {
            circle(trackImage, Point(pointsEdgeRight[i].y, pointsEdgeRight[i].x), 1,
                Scalar(0, 255, 255), -1); // 黄色点
        }

        for (int i = 0; i < inlines.size(); i++)
        {
            circle(trackImage, Point(inlines[i].y, inlines[i].x), 3,
                Scalar(0, 0, 255), -1); // 红色点
        }
        for (int i = 0; i < corners.size(); i++)
        {
            circle(trackImage, Point(corners[i].y, corners[i].x), 3,
                Scalar(178, 102, 255), -1); //粉色点
        }
    circle(trackImage, Point(BottomPoint.y, BottomPoint.x), 7, Scalar(255, 102, 178), 1);
    int blocks = GetSlope0Bottom(0);
    putText(trackImage, to_string(blocks), Point(140, 180), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
    int blocks_r = GetSlope0Bottom(1);
    putText(trackImage, to_string(blocks_r), Point(180, 180), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
    int index = GetEndIndex(0);
    putText(trackImage, to_string(index), Point(20, 200), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    index = GetEndIndex(1);
    putText(trackImage, to_string(index), Point(270, 200), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    index = GetMiddleIndex(0);
    putText(trackImage, to_string(index), Point(20, 190), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    index = GetMiddleIndex(1);
    putText(trackImage, to_string(index), Point(270, 190), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    putText(trackImage, 'g'+to_string(garageEnable.x), Point(100, 190), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

    }



    /**
     * @brief 边缘方差计算
     *
     * @param v_edge
     * @param img_height
     * @return double
     */
    double stdevEdgeCal(vector<POINT> &v_edge, int img_height)
    {
        if (v_edge.size() < img_height / 4)
        {
            return 1000;
        }
        vector<int> v_slope;
        int step = 10; // v_edge.size()/10;
        for (int i = step; i < v_edge.size(); i += step)
        {
            if (v_edge[i].x - v_edge[i - step].x)
                v_slope.push_back((v_edge[i].y - v_edge[i - step].y) * 100 / (v_edge[i].x - v_edge[i - step].x));
        }
        if (v_slope.size() > 1)
        {
            double sum = accumulate(begin(v_slope), end(v_slope), 0.0);
            double mean = sum / v_slope.size(); // 均值
            double accum = 0.0;
            for_each(begin(v_slope), end(v_slope), [&](const double d)
                     { accum += (d - mean) * (d - mean); });

            return sqrt(accum / (v_slope.size() - 1)); // 方差
        }
        else
            return 0;
    }
    
    double stdevEdgeCal50(vector<POINT> &v_edge, int img_height)
    {
        vector<int> v_slope;
        int step = 10; // v_edge.size()/10;
        for (int i = step; i < v_edge.size(); i += step)
        {
            if (v_edge[i].x - v_edge[i - step].x)
                v_slope.push_back((v_edge[i].y - v_edge[i - step].y) * 100 / (v_edge[i].x - v_edge[i - step].x));
        }
        if (v_slope.size() > 1)
        {
            double sum = accumulate(begin(v_slope), end(v_slope), 0.0);
            double mean = sum / v_slope.size(); // 均值
            double accum = 0.0;
            for_each(begin(v_slope), end(v_slope), [&](const double d)
                     { accum += (d - mean) * (d - mean); });

            return sqrt(accum / (v_slope.size() - 1)); // 方差
        }
        else
            return 0;
    }
    // -2 全部点集为边际 -1 前半部分无边际 返回正数前半部分为边际  适合做判断算子
    int GetEndIndex(int right) {
        int index = -1;
        if (right != 0) {
            for (int i = 0; i < pointsEdgeRight.size(); i++) {
                if (pointsEdgeRight[i].y > 317) {
                    index = i;
                    
                }
                else if (pointsEdgeRight[i].y < 280) {
                    return index;
                }
            }
        } 
        else {
            for (int i = 0; i < pointsEdgeLeft.size(); i++) {
                if (pointsEdgeLeft[i].y < 2) {
                    index = i;
                }
                else if (pointsEdgeLeft[i].y > 30) {
                    return index;
                }
            }
        }
        if (index == -1) {
            return -1;
        }
        else
            return -2;
    }
    //-1 全集无边际 会提取大于30宽度的最近边际 适合用作补线算子
    int GetMiddleIndex(int right) {
        int index = -1;
        int counter = 0;
        int threshold = 30;

        if (right != 0) {
            for (int i = 0; i < pointsEdgeRight.size(); i++) {
                if (pointsEdgeRight[i].y > 317) {
                    index = i;
                    counter++;
                }
                else if (counter != 0) {
                    if (counter > threshold)return index;
                    counter = 0;
                    index = -1;
                }
            }
        }
        else {
            for (int i = 0; i < pointsEdgeLeft.size(); i++) {
                if (pointsEdgeLeft[i].y < 2) {
                    index = i;
                    counter++;
                }
                else if (counter != 0) {
                    if (counter > threshold)return index;
                    counter == 0;
                }
            }
        }
        if (threshold > 10)
            return index;
        else
            return -1;
    }


    int GetSlope0Bottom(int right) {
        int counter = 0;
        int rows = 120;
        if (right) {
            if (pointsEdgeRight.size() < rows)return -1;
            for (int i = 0; i < rows; i++) {
                if (pointsEdgeRight[i].slope == 0)
                    counter++;
            }
        }
        else {
            if (pointsEdgeLeft.size() < rows)return -1;
            for (int i = 0; i < rows; i++) {
                if (pointsEdgeLeft[i].slope == 0)
                    counter++;
            }
        }
        return counter;

    }
    void AdjustbyParking(int right){
        if (right && !pointsEdgeRight.empty()) {  // 增加非空检查
            vector<POINT> tmp_pointsEdgeRight;
            tmp_pointsEdgeRight.push_back(pointsEdgeRight[0]);  // 先保存第一个点
            int tmp_y = pointsEdgeRight[0].y;

            // 遍历所有元素（修正循环条件）
            for (size_t i = 1; i < pointsEdgeRight.size(); i++) {
                if (pointsEdgeRight[i].y <= tmp_y) {  // 保留y值更小的点
                    tmp_pointsEdgeRight.push_back(pointsEdgeRight[i]);
                    tmp_y = pointsEdgeRight[i].y;
                }
            }
            pointsEdgeRight = tmp_pointsEdgeRight;  // 更新原始数组
        }else if(!right && !pointsEdgeLeft.empty()){
            vector<POINT> tmp_pointsEdgeleft;
            tmp_pointsEdgeleft.push_back(pointsEdgeLeft[0]);  // 先保存第一个点
            int tmp_y = pointsEdgeLeft[0].y;

            // 遍历所有元素（修正循环条件）
            for (size_t i = 1; i < pointsEdgeLeft.size(); i++) {
                if (pointsEdgeLeft[i].y >= tmp_y) {  // 保留y值更小的点
                    tmp_pointsEdgeleft.push_back(pointsEdgeLeft[i]);
                    tmp_y = pointsEdgeLeft[i].y;
                }
            }
            pointsEdgeLeft = tmp_pointsEdgeleft;  // 更新原始数组
        }
    }
    int CalculateRoiAverageGray(const POINT& roiPoint, int width=30 , int hight=80 ) {
        // roiPoint 为下边的中点
        mean = 0;
        hight = roiPoint.x / 3;
        int col = roiPoint.y - width/2;
        int row = roiPoint.x - hight;

        // 创建ROI矩形（确保不超出图像边界）
        cv::Rect tmp(
            std::max(0, col),               // x坐标
            std::max(0, row),               // y坐标
            std::min(width, 319 - col),           // 宽度
            std::min(hight, 239 - row)         // 高度（确保不超出图像底部）
        );
        box_roi = tmp;
        // 提取ROI并计算平均灰度值
        cv::Mat roiImage = imagePath(box_roi);
        cv::Scalar meanValue = cv::mean(roiImage);
        mean = meanValue[0];
        return meanValue[0];  // 返回单通道图像的平均灰度值
    }

    void DrawRect(Mat& Image) {
        putText(Image, to_string(mean), Point(160, 200), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        cv::rectangle(Image, box_roi, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

void GetBlocksNum()
{
    if(block.size() > 1)
    {
        block_num ++;
    }
    else 
    {
        block_one ++;
        if(block_one >= 2)
        {
            if(block_num >= 4)
            {
            //block_num = 0;
                blocks_num ++;
                pointsBlockFar.push_back((pointsEdgeLeft[pointsEdgeLeft.size() - 1].x + pointsEdgeRight[pointsEdgeRight.size() - 1].x) / 2);
            }
            block_num = 0;
            block_one = 0;        
        }
        else
        {
            block_num ++;
        }       
    }
}

private:
    Mat imagePath; // 赛道搜索图像
    vector<POINT>block; 
    int max_width; // 一行中最大色块
    int indexWidestBlock; // 最宽色块索引
    Mat k_e = getStructuringElement(MORPH_RECT, Size(3, 3)); //腐蚀核
    Mat k_d = getStructuringElement(MORPH_RECT, Size(3, 3)); //膨胀核
    int  mean = 0;
    void FilterbyDE(Mat& img) {
      cv::dilate(img, img, k_d, cv::Point(-1, -1), 1); //膨胀
      cv::erode(img, img, k_e, cv::Point(-1, -1), 1);  //腐蚀
    }
    /**
     * @brief 边缘斜率计算
     *
     * @param edge
     * @param index
     */
    void slopeCal(vector<POINT> &edge, int index)
    {
        if (index <= 4)
        {
            return;
        }
        float temp_slop1 = 0.0, temp_slop2 = 0.0;
        if (edge[index].x - edge[index - 2].x != 0)
        {
            temp_slop1 = (float)(edge[index].y - edge[index - 2].y) * 1.0f /
                         ((edge[index].x - edge[index - 2].x) * 1.0f);
        }
        else
        {
            temp_slop1 = edge[index].y > edge[index - 2].y ? 255 : -255;
        }
        if (edge[index].x - edge[index - 4].x != 0)
        {
            temp_slop2 = (float)(edge[index].y - edge[index - 4].y) * 1.0f /
                         ((edge[index].x - edge[index - 4].x) * 1.0f);
        }
        else
        {
            edge[index].slope = edge[index].y > edge[index - 4].y ? 255 : -255;
        }
        if (abs(temp_slop1) != 255 && abs(temp_slop2) != 255)
        {
            edge[index].slope = (temp_slop1 + temp_slop2) * 1.0 / 2;
        }
        else if (abs(temp_slop1) != 255)
        {
            edge[index].slope = temp_slop1;
        }
        else
        {
            edge[index].slope = temp_slop2;
        }
    }


    /**
     * @brief 冒泡法求取集合中值
     *
     * @param vec 输入集合
     * @return int 中值
     */
    int getMiddleValue(vector<int> vec)
    {
        if (vec.size() < 1)
            return -1;
        if (vec.size() == 1)
            return vec[0];

        int len = vec.size();
        while (len > 0)
        {
            bool sort = true; // 是否进行排序操作标志
            for (int i = 0; i < len - 1; ++i)
            {
                if (vec[i] > vec[i + 1])
                {
                    swap(vec[i], vec[i + 1]);
                    sort = false;
                }
            }
            if (sort) // 排序完成
                break;

            --len;
        }

        return vec[(int)vec.size() / 2];
    }
};

