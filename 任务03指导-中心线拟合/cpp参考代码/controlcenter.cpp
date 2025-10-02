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
 * @file controlcenter.cpp
 * @author Leo
 * @brief 智能车控制中心计算
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
#include "../include/common.hpp"
#include "recognition/tracking.cpp"
#include <algorithm>
using namespace cv;
using namespace std;

extern uint16_t turnError;
extern uint16_t turnRow;
extern uint16_t turnTimeControl;
extern uint8_t LU_flag;

class ControlCenter
{
public:
    bool corner_state = true;
    int controlCenter;           // 智能车控制中心（0~320）
    int contrlCenter40;
    int contrlCenterbottom;
    vector<POINT> centerEdge;    // 赛道中心点集
    uint16_t validRowsLeft = 0;  // 边缘有效行数（左）
    uint16_t validRowsRight = 0; // 边缘有效行数（右）
    double sigmaCenter = 0;      // 中心点集的方差
    int error = 0;
    int error40 = 0;
    int errorbottom = 0;
    int index = 0;

    float errorbottom_norm = 0.0;
    float error_norm = 0.0;

    float errorbottom_norm_standard = 40; //15.0; 
    float error_norm_standard =  20; // 40.0;

    /**
     * @brief 控制中心计算
     *
     * @param pointsEdgeLeft 赛道左边缘点集
     * @param pointsEdgeRight 赛道右边缘点集
     * return centerEdge（中心点集）
     *         ->加权处理：controlcentre,控制中心
     *         ->方差处理：sigmaCenter,控制率
     * 
     */

    void fitting(Tracking &track)
    {
        sigmaCenter = 0;
        controlCenter = COLSIMAGE / 2;
        contrlCenterbottom = 0;
        contrlCenter40 = 0;
        centerEdge.clear();
        vector<POINT> v_center(4); // 三阶贝塞尔曲线
        style = "STRAIGHT";
        int leftindex = 0;
        int rightindex = 0;
        for (int i = track.pointsEdgeLeft.size() - 1; i > 1; i--) {
            if (track.pointsEdgeLeft[i].x > topcut) {
                leftindex = i;
                break;
            }
        }
        for (int i = track.pointsEdgeRight.size() - 1; i > 1; i--) {
            if (track.pointsEdgeRight[i].x > topcut) {
                rightindex = i;
                break;
            }
        }
        // 边缘斜率重计算（边缘修正之后）
        track.stdevLeft = track.stdevEdgeCal(track.pointsEdgeLeft, ROWSIMAGE);
        track.stdevRight = track.stdevEdgeCal(track.pointsEdgeRight, ROWSIMAGE);


        if (track.pointsEdgeLeft.size() > 4 && track.pointsEdgeRight.size() > 4) // 通过双边缘有效点的差来判断赛道类型,4等分拟合，所以num_edge>4
        {

            v_center[0] = { (track.pointsEdgeLeft[0].x + track.pointsEdgeRight[0].x) / 2, (track.pointsEdgeLeft[0].y + track.pointsEdgeRight[0].y) / 2 };

            v_center[1] = { (track.pointsEdgeLeft[leftindex / 3].x + track.pointsEdgeRight[rightindex / 3].x) / 2,
                        (track.pointsEdgeLeft[leftindex / 3].y + track.pointsEdgeRight[rightindex / 3].y) / 2 };

            v_center[2] = { (track.pointsEdgeLeft[leftindex * 2 / 3].x + track.pointsEdgeRight[rightindex * 2 / 3].x) / 2,
                        (track.pointsEdgeLeft[leftindex * 2 / 3].y + track.pointsEdgeRight[rightindex * 2 / 3].y) / 2 };
            if (track.pointsEdgeLeft[leftindex].y > track.pointsEdgeRight[rightindex].y)
                track.pointsEdgeLeft[leftindex].y = track.pointsEdgeRight[rightindex].y;
            v_center[3] = { (track.pointsEdgeLeft[leftindex].x + track.pointsEdgeRight[rightindex].x) / 2,
                        (track.pointsEdgeLeft[leftindex].y + track.pointsEdgeRight[rightindex].y) / 2 };

            centerEdge = Bezier(0.03, v_center);

            style = "STRAIGHT";
        }
        // 左单边
        else if ((track.pointsEdgeLeft.size() > 0 && track.pointsEdgeRight.size() <= 4) ||
            (track.pointsEdgeLeft.size() > 0 && track.pointsEdgeRight.size() > 0 && track.pointsEdgeLeft[0].x - track.pointsEdgeRight[0].x > ROWSIMAGE / 2))
        {
            style = "RIGHT";
            centerEdge = centerCompute(track.pointsEdgeLeft, 0);
        }
        // 右单边
        else if ((track.pointsEdgeRight.size() > 0 && track.pointsEdgeLeft.size() <= 4) ||
            (track.pointsEdgeRight.size() > 0 && track.pointsEdgeLeft.size() > 0 && track.pointsEdgeRight[0].x - track.pointsEdgeLeft[0].x > ROWSIMAGE / 2))
        {
            style = "LEFT";
            centerEdge = centerCompute(track.pointsEdgeRight, 1);
        }
        else if (track.pointsEdgeLeft.size() > 4 && track.pointsEdgeRight.size() == 0) // 左单边
        {
            v_center[0] = { track.pointsEdgeLeft[0].x, (track.pointsEdgeLeft[0].y + COLSIMAGE - 1) / 2 };

            v_center[1] = { track.pointsEdgeLeft[leftindex / 3].x,
                        (track.pointsEdgeLeft[leftindex / 3].y + COLSIMAGE - 1) / 2 };

            v_center[2] = { track.pointsEdgeLeft[leftindex * 2 / 3].x,
                        (track.pointsEdgeLeft[leftindex * 2 / 3].y + COLSIMAGE - 1) / 2 };

            v_center[3] = { track.pointsEdgeLeft[leftindex].x,
                        (track.pointsEdgeLeft[leftindex].y + COLSIMAGE - 1) / 2 };

            centerEdge = Bezier(0.02, v_center);

            style = "RIGHT";
        }
        else if (track.pointsEdgeLeft.size() == 0 && track.pointsEdgeRight.size() > 4) // 右单边
        {
            v_center[0] = { track.pointsEdgeRight[0].x, track.pointsEdgeRight[0].y / 2 };

            v_center[1] = { track.pointsEdgeRight[rightindex / 3].x,
                        track.pointsEdgeRight[rightindex / 3].y / 2 };

            v_center[2] = { track.pointsEdgeRight[rightindex * 2 / 3].x,
                        track.pointsEdgeRight[rightindex * 2 / 3].y / 2 };

            v_center[3] = { track.pointsEdgeRight[rightindex].x,
                        track.pointsEdgeRight[rightindex].y / 2 };

            centerEdge = Bezier(0.02, v_center);

            style = "LEFT";
        }
            // 加权控制中心计算

        int controlNum = 1;
        for(auto p : centerEdge)
        {
            if(abs(p.y-COLSIMAGE/2) < turnError && p.x > turnRow )//corner_state false 为近端直道
                {
                    corner_state = false;
                }
        }
        for (auto p : centerEdge)
        {
            if(corner_state)//
            {
                if(p.x < turnRow)//近端弯道模式
                {
                    controlNum++;
                    controlCenter += p.y;
                }
            }
            else //近端直线模式
            {
                if (p.x > turnTimeControl)//分段
                {
                controlNum += turnTimeControl;
                controlCenter += p.y * turnTimeControl;
                }
                else
                {
                    controlNum += p.x/2 + turnTimeControl/2;
                    controlCenter += p.y * ( p.x/2 + turnTimeControl/2);
                }
            }
        }
        if (controlNum > 1)
        {
            controlCenter = controlCenter / controlNum;
        }

        if (controlCenter > COLSIMAGE)
            controlCenter = COLSIMAGE;
        else if (controlCenter < 0)
            controlCenter = 0;
        // std::cout << "Corner_state value: " << corner_state << std::endl;
        ////////////////////////////// the final value of error ////////////////////////////////
        error = controlCenter-COLSIMAGE/2;
        error_norm = error / error_norm_standard;
        // error = error_norm;
        ////////////////////////////// the final value of error ////////////////////////////////

        //100-140的error
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        index = 0;
        //找140行
        for(int i = 40;i<track.pointsEdgeLeft.size();i++) 
        {
            if(track.pointsEdgeLeft[i].x <= ROWSIMAGE/2+20)
            {
                index = i;
                break;
            }
        }
        if(index != 0)
        {
            //求平均值
            int num = track.pointsEdgeLeft.size();
            int endindex =  min(index+40,num-1);
            for(int i = index; i< endindex;i++)
            {
                contrlCenter40 += (track.pointsEdgeRight[i].y+track.pointsEdgeLeft[i].y)/2;
                //printf("index: %d y:%d\n",index,centerEdge[i].y);
            }
            if(endindex-index != 0)
            {
                contrlCenter40 = contrlCenter40 / (endindex-index);
                error40 = contrlCenter40 - COLSIMAGE/2;
                //printf("error:%d , error40:%d\n",error,error40);
            }
            /*vector<POINT> centerV40;
            for(int i = 0; i < index;i++)
            {
                centerV40.push_back(centerEdge[i]);
            }
            sigmaCenterbottom = sigma(centerV40);
            //printf("%.2f\n",sigmaCenterbottom);  */     
        }
                //100-140的error

            
       //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //bottom
            //求平均值
            for(int i = 0; i< 40;i++)
            {
                contrlCenterbottom += (track.pointsEdgeRight[i].y+track.pointsEdgeLeft[i].y)/2;
                //printf("index: %d y:%d\n",index,centerEdge[i].y);
            }
            contrlCenterbottom = contrlCenterbottom /40;
            ////////////////////////////// the final value of erbt ////////////////////////////////
            errorbottom = contrlCenterbottom - COLSIMAGE/2;
            errorbottom_norm = (float)errorbottom / errorbottom_norm_standard;
            // errorbottom = errorbottom_norm;
            ////////////////////////////// the final value of erbt ////////////////////////////////

        // 控制率计算
        if (centerEdge.size() > 20)
        {
            vector<POINT> centerV;
            int filt = centerEdge.size() / 5;
            for (int i = filt; i < centerEdge.size() - filt; i++) // 过滤中心点集前后1/5的诱导性
            {
                centerV.push_back(centerEdge[i]);
            }
            sigmaCenter = sigma(centerV);
        }
        else
            sigmaCenter = 1000;


    }

    /**
     * @brief 车辆冲出赛道检测（保护车辆）
     *
     * @param track
     * @return true
     * @return false
     */
    bool derailmentCheck(Tracking track)
    {
        if (track.pointsEdgeLeft.size() < 30 && track.pointsEdgeRight.size() < 30) // 防止车辆冲出赛道
        {
            countOutlineA++;
            countOutlineB = 0;
            if (countOutlineA > 20)
                return true;
        }
        else
        {
            countOutlineB++;
            if (countOutlineB > 50)
            {
                countOutlineA = 0;
                countOutlineB = 50;
            }
        }
        return false;
    }

    /**
     * @brief 显示赛道线识别结果
     *
     * @param centerImage 需要叠加显示的图像
     */
    void drawImage(Tracking track, Mat &centerImage)
    {
        // 赛道边缘绘制
        for (int i = 0; i < track.pointsEdgeLeft.size(); i++)
        {
            circle(centerImage, Point(track.pointsEdgeLeft[i].y, track.pointsEdgeLeft[i].x), 1, Scalar(0, 255, 0), -1); // 绿色点
        }
        for (int i = 0; i < track.pointsEdgeRight.size(); i++)
        {
            circle(centerImage, Point(track.pointsEdgeRight[i].y, track.pointsEdgeRight[i].x), 1, Scalar(255, 255, 0), -1); // 黄色点
        }
        for (int i = 0; i < track.corners.size(); i++)
        {
            circle(centerImage, Point(track.corners[i].y, track.corners[i].x), 3,
                Scalar(0, 0, 255), -1); // 红色点
        }
        for (int i = 0; i < track.inlines.size(); i++)
        {
            circle(centerImage, Point(track.inlines[i].y, track.inlines[i].x), 3,
                 Scalar(178, 102, 255), -1); //粉色点
        }

        // 绘制中心点集
        for (int i = 0; i < centerEdge.size(); i++)
        {
            circle(centerImage, Point(centerEdge[i].y, centerEdge[i].x), 1, Scalar(0, 0, 255), -1);
        }

        /*for(int i = 0;i < 319; i++ )
        {
            circle(centerImage, Point(i, 100),1, Scalar(0, 255, 0), -1);
            circle(centerImage, Point(i, 140),1, Scalar(0, 255, 0), -1);
        }*/

        // 绘制加权控制中心：方向
        //Rect rect(controlCenter, ROWSIMAGE - 20, 10, 20);
        //rectangle(centerImage, rect, Scalar(0, 0, 255), CV_FILLED);

        // 详细控制参数显示
        int dis = 20;
        string str;
       // putText(centerImage, style, Point(COLSIMAGE - 60, dis), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255), 1); // 赛道类型

        str = "Edge: " + formatDoble2String(track.stdevLeft, 1) + " | " + formatDoble2String(track.stdevRight, 1);
        putText(centerImage, str, Point(COLSIMAGE - 150, 2 * dis), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255), 1); // 斜率：左|右

        str = "Center: " + formatDoble2String(sigmaCenter, 2);
        putText(centerImage, str, Point(COLSIMAGE - 120, 3 * dis), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255), 1); // 中心点方差
        //putText(centerImage, to_string(contrlCenter40), Point(COLSIMAGE - 120, 4 * dis), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0), 1); // 中心点方差
        putText(centerImage, to_string(errorbottom), Point(COLSIMAGE / 2 - 40, ROWSIMAGE - 20), FONT_HERSHEY_PLAIN, 1.2, Scalar(255, 0, 255), 1); // 中心
        putText(centerImage, to_string(errorbottom_norm), Point(COLSIMAGE / 2 + 40, ROWSIMAGE - 20), FONT_HERSHEY_PLAIN, 1.2, Scalar(255, 0, 255), 1); // 中心
        putText(centerImage, to_string(error), Point(COLSIMAGE / 2 - 40, ROWSIMAGE - 40), FONT_HERSHEY_PLAIN, 1.2, Scalar(0, 0, 255), 1); // 中心
        putText(centerImage, to_string(error_norm), Point(COLSIMAGE / 2 + 40, ROWSIMAGE - 40), FONT_HERSHEY_PLAIN, 1.2, Scalar(0, 0, 255), 1); // 中心
        //putText(centerImage, to_string(error40), Point(COLSIMAGE / 2 - 10, ROWSIMAGE - 60), FONT_HERSHEY_PLAIN, 1.2, Scalar(0, 255, 0), 1); // 中心
        putText(centerImage, "LU_flag" + to_string(LU_flag), Point(COLSIMAGE / 2 - 10, ROWSIMAGE/2), FONT_HERSHEY_PLAIN, 1.2, Scalar(0, 0, 255), 1); // 中心
        //putText(centerImage, "cross_step" + to_string(crossroad), Point(COLSIMAGE / 2 - 10, ROWSIMAGE/2-30), FONT_HERSHEY_PLAIN, 1.2, Scalar(0, 0, 255), 1);
        //putText(centerImage, "resuce_step" + to_string(rescue), Point(COLSIMAGE / 2 - 30, ROWSIMAGE/2+60), FONT_HERSHEY_PLAIN, 1.2, Scalar(0, 0, 255), 1);         
    }

private:
    int countOutlineA = 0; // 车辆脱轨检测计数器
    int countOutlineB = 0; // 车辆脱轨检测计数器
    string style = "";     // 赛道类型
    int topcut = 50;
    /**
     * @brief 搜索十字赛道突变行（左下）
     *
     * @param pointsEdgeLeft
     * @return uint16_t
     */
    uint16_t searchBreakLeftDown(vector<POINT> pointsEdgeLeft)//寻找y=2的向中突变
    {
        uint16_t counter = 0;

        for (int i = 0; i < pointsEdgeLeft.size() - 10; i++)
        {
            if (pointsEdgeLeft[i].y >= 2)
            {
                counter++;
                if (counter > 3)
                {
                    return i - 2;
                }
            }
            else
                counter = 0;
        }

        return 0;
    }

    /**
     * @brief 搜索十字赛道突变行（右下）
     *
     * @param pointsEdgeRight
     * @return uint16_t
     */
    uint16_t searchBreakRightDown(vector<POINT> pointsEdgeRight)
    {
        uint16_t counter = 0;

        for (int i = 0; i < pointsEdgeRight.size() - 10; i++) // 寻找左边跳变点
        {
            if (pointsEdgeRight[i].y < COLSIMAGE - 2)
            {
                counter++;
                if (counter > 3)
                {
                    return i - 2;
                }
            }
            else
                counter = 0;
        }

        return 0;
    }

    /**
     * @brief 赛道中心点计算：单边控制
     *
     * @param pointsEdge 赛道边缘点集
     * @param side 单边类型：左边0/右边1
     * @return vector<POINT>
     */
    vector<POINT> centerCompute(vector<POINT> pointsEdge, int side)
    {
        int step = 4;                    // 间隔尺度
        int offsetWidth = COLSIMAGE / 2; // 首行偏移量
        int offsetHeight = 0;            // 纵向偏移量

        vector<POINT> center; // 控制中心集合

        if (side == 0) // 左边缘
        {
            uint16_t counter = 0, rowStart = 0;
            for (int i = 0; i < pointsEdge.size(); i++) // 删除底部无效行
            {
                if (pointsEdge[i].y > 1)
                {
                    counter++;
                    if (counter > 2)
                    {
                        rowStart = i - 2;
                        break;
                    }
                }
                else
                    counter = 0;
            }

            offsetHeight = pointsEdge[rowStart].x - pointsEdge[0].x;
            counter = 0;
            for (int i = rowStart; i < pointsEdge.size(); i += step)
            {
                int py = pointsEdge[i].y + offsetWidth;
                if (py > COLSIMAGE - 1)
                {
                    counter++;
                    if (counter > 2)
                        break;
                }
                else
                {
                    counter = 0;
                    center.emplace_back(pointsEdge[i].x - offsetHeight, py);
                }
            }
        }
        else if (side == 1) // 右边沿
        {
            uint16_t counter = 0, rowStart = 0;
            for (int i = 0; i < pointsEdge.size(); i++) // 删除底部无效行
            {
                if (pointsEdge[i].y < COLSIMAGE - 1)
                {
                    counter++;
                    if (counter > 2)
                    {
                        rowStart = i - 2;
                        break;
                    }
                }
                else
                    counter = 0;
            }

            offsetHeight = pointsEdge[rowStart].x - pointsEdge[0].x;
            counter = 0;
            for (int i = rowStart; i < pointsEdge.size(); i += step)
            {
                int py = pointsEdge[i].y - offsetWidth;
                if (py < 1)
                {
                    counter++;
                    if (counter > 2)
                        break;
                }
                else
                {
                    counter = 0;
                    center.emplace_back(pointsEdge[i].x - offsetHeight, py);
                }
            }
        }

        return center;
        // return Bezier(0.2,center);
    }
};