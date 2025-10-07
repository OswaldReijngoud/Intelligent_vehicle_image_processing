#pragma once


#include <fstream>
#include <iostream>
#include <cmath>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "../../include/common.hpp"
#include "tracking.cpp"


using namespace cv;
using namespace std;

class Crossroad
{
public:
	enum CrossStep{
		None=0,
		Fix,
	};
	enum CrossType {
		Left = 0,
	    Right,
		Mid
	};
	CrossStep step = None;
	CrossType type = Left;
	POINT TopPoint;
	POINT LeftBottomPoint;
	POINT RightBottomPoint;
	bool process(Tracking& track);
	void draw_img(Mat& img);
	int FindBottomPoint(Tracking& track, int right);
	int FindBottomPoint2(Tracking& track, int right);
	int WholeBlock(Tracking track);
private:
	POINT startpoint = POINT(0, 0);
	POINT midpoint = POINT(0, 0);
	POINT endpoint = POINT(0, 0);
	int width_counter = 0;
	int memory = 0;
	vector<POINT> bezier_tmp;
	bool right_flag = false;
	bool left_flag = false;
};


bool Crossroad::process(Tracking& track) {
	switch (step) {
		case CrossStep::None: {
			if (!track.inlines.empty()) {
				int bottom_right_index = FindBottomPoint(track, 1);
				int bottom_left_index = FindBottomPoint(track, 0);
				if (right_flag || left_flag) {// CASE 1 
					int left_endindex = 0;
					int right_endindex = 0;
					int left_mid = 0;
					int right_mid = 0;
					left_endindex = track.GetEndIndex(0);
					right_endindex = track.GetEndIndex(1);
					left_mid = track.GetMiddleIndex(0);
					right_mid = track.GetMiddleIndex(1);
					if (right_flag && (left_endindex > 100 || left_mid > 100) && right_endindex < 60 && track.inlines[0].y < 100 && track.inlines[0].x < 120) {//右斜入
						LOGLN("RIGHT CASE 1");
						memory = track.inlines[0].y;
						type = CrossType::Right;
						step = CrossStep::Fix;
					}
					else if (left_flag && (right_endindex > 100 || right_mid > 100) && left_endindex < 60 && track.inlines[0].y > 220 && track.inlines[0].x < 120) {//左斜入
						LOGLN("LEFT CASE 1");
						memory = track.inlines[0].y;
						type = CrossType::Left;
						step = CrossStep::Fix;
					}
					else if (left_mid > 140 && right_mid > 140 && abs(track.stdevLeft - track.stdevRight) < 100) {//直入
						LOGLN("MID CASE 1");
						type = CrossType::Mid;
						step = CrossStep::Fix;
					}
				}

			}
			break;
		}
		case CrossStep::Fix: {
			int blockline = WholeBlock(track);
			if(type == CrossType::Right){
				if (!track.inlines.empty()) {
					int bottom_right_index = FindBottomPoint2(track, 1);
					if (bottom_right_index == -1) {
						track.pointsEdgeRight.clear();
						track.pointsEdgeRight = bezier_tmp;
						track.pointsEdgeLeft.resize(track.pointsEdgeRight.size());
					}
					else {
						if (memory - track.inlines[0].y > 50)
							track.inlines[0].y = COLSIMAGE - track.inlines[0].y;
						startpoint = track.pointsEdgeRight[0];
						endpoint = track.inlines[0];
						midpoint = track.pointsEdgeRight[bottom_right_index];
						vector<POINT> input = { startpoint, midpoint, endpoint };
						bezier_tmp.clear();
						bezier_tmp = Bezier(0.01, input);
						track.pointsEdgeRight.clear();
						track.pointsEdgeRight = bezier_tmp;
						track.pointsEdgeLeft.resize(track.pointsEdgeRight.size());
					}
					memory = track.inlines[0].y;
					if (track.inlines[0].y > 220 || width_counter > 40) {
						LOGLN("RIGHT EXIT:"<<track.inlines[0].y<<' '<<width_counter);
						step = CrossStep::None;
					}
				}
			}
			else if(type == CrossType::Left){
				if (!track.inlines.empty()) {
					int bottom_left_index = FindBottomPoint2(track, 0);
					if (bottom_left_index == -1) {
						track.pointsEdgeLeft.clear();
						track.pointsEdgeLeft = bezier_tmp;
						track.pointsEdgeRight.resize(track.pointsEdgeLeft.size());
					}
					else {
						if (memory - track.inlines[0].y < -50)
							track.inlines[0].y = COLSIMAGE - track.inlines[0].y;
						startpoint = track.pointsEdgeLeft[0];
						endpoint = track.inlines[0];
						midpoint = track.pointsEdgeLeft[bottom_left_index];
						LOGLN(bottom_left_index);
						vector<POINT> input = { startpoint, midpoint, endpoint };
						bezier_tmp.clear();
						bezier_tmp = Bezier(0.01, input);
						track.pointsEdgeLeft.clear();
						track.pointsEdgeLeft = bezier_tmp;
						track.pointsEdgeRight.resize(track.pointsEdgeLeft.size());
					}
					memory = track.inlines[0].y;
					if (track.inlines[0].y < 100 || width_counter > 40) {
						LOGLN("LEFT EXIT:"<<track.inlines[0].y<<' '<<width_counter);
						step = CrossStep::None;
					}
				}
			}
			else {
				if (blockline == -1)return;
				int left_mid = 0;
				int right_mid = 0;
				left_mid = track.GetMiddleIndex(0);
				right_mid = track.GetMiddleIndex(1);
				if ((left_mid < 60 && right_mid < 60) || track.widthBlock[blockline].x > 160) {
					LOGLN("MID EXIT:"<<' '<<track.widthBlock[blockline].x);
					step = CrossStep::None;
				}
			}
			break;
		}
	}

}
void Crossroad::draw_img(Mat& img) {
	circle(img, Point(RightBottomPoint.y, RightBottomPoint.x), 3, Scalar(255, 106, 106), -1); //红色点
	circle(img, Point(LeftBottomPoint.y, LeftBottomPoint.x), 3, Scalar(255, 255, 106), -1); //红色点
	putText(img, "width " + to_string(width_counter), Point(240, 220), cv::FONT_HERSHEY_TRIPLEX, 0.3, cv::Scalar(255, 106, 106), 1, cv::LINE_AA);
	putText(img, "flag " + to_string(right_flag), Point(220, 200), cv::FONT_HERSHEY_TRIPLEX, 0.3, cv::Scalar(255, 106, 106), 1, cv::LINE_AA);
	putText(img, "flag " + to_string(left_flag), Point(100, 200), cv::FONT_HERSHEY_TRIPLEX, 0.3, cv::Scalar(255, 106, 106), 1, cv::LINE_AA);
	putText(img, "step "+to_string(step), Point(160, 200), cv::FONT_HERSHEY_TRIPLEX, 0.3, cv::Scalar(255, 106, 106), 1, cv::LINE_AA);
	putText(img, "type " + to_string(type), Point(160, 220), cv::FONT_HERSHEY_TRIPLEX, 0.3, cv::Scalar(255, 106, 106), 1, cv::LINE_AA);
}
int Crossroad::FindBottomPoint(Tracking& track,int right) {
	int index = 0;
	int col = 0;
	int counter = 0;
	if (right) {
		RightBottomPoint = POINT(0, 0);
		col = track.pointsEdgeRight[0].y;
		for (int i = 1; i < track.pointsEdgeRight.size(); i++) {
			if (track.pointsEdgeRight[i].y < col) {
				col = track.pointsEdgeRight[i].y;
				index = i;
			}
			else if (track.pointsEdgeRight[i].y > col + 5 && track.pointsEdgeRight[i].y < col + 50) {
				RightBottomPoint = track.pointsEdgeRight[index];
				col = track.pointsEdgeRight[i].y;
				index = i;
				break;
			}
			else if (track.pointsEdgeRight[i].y >= col + 50) {
				right_flag = false;
				return -1;
			}
		}
		for (int i = index; i < track.pointsEdgeRight.size(); i++) {
			if (track.pointsEdgeRight[i].y > col) {
				col = track.pointsEdgeRight[i].y;
				counter++;
				if (counter > 15) {
					right_flag = true;
					return index;
				}
			}
		}
		return -1;
	}
	else {
		LeftBottomPoint = POINT(0, 0);
		col = track.pointsEdgeLeft[0].y;
		for (int i = 1; i < track.pointsEdgeLeft.size(); i++) {
			if (track.pointsEdgeLeft[i].y > col) {
				col = track.pointsEdgeLeft[i].y;
				index = i;
			}
			else if (track.pointsEdgeLeft[i].y < col - 5 && track.pointsEdgeLeft[i].y > col - 50) {
				LeftBottomPoint = track.pointsEdgeLeft[index];
				col = track.pointsEdgeLeft[i].y;
				index = i;
				break;
			}
			else if (track.pointsEdgeLeft[i].y <= col - 50) {
				left_flag = false;
				return -1;
			}
		}
		for (int i = index; i < track.pointsEdgeLeft.size(); i++) {
			if (track.pointsEdgeLeft[i].y < col) {
				col = track.pointsEdgeLeft[i].y;
				counter++;
				if (counter > 15) {
					left_flag = true;
					return index;
				}
			}
		}
		return -1;
	}
	return index;
}
int Crossroad::FindBottomPoint2(Tracking& track, int right) {
	int index = 0;
	int col = 0;
	if (right) {
		col = track.pointsEdgeRight[0].y;
		for (int i = 1; i < track.pointsEdgeRight.size(); i++) {
			if (track.pointsEdgeRight[i].y <= col) {
				col = track.pointsEdgeRight[i].y;
				index = i;
			}
			else {
				return i;
			}
		}
	}
	else {
		col = track.pointsEdgeLeft[0].y;
		for (int i = 1; i < track.pointsEdgeLeft.size(); i++) {
			if (track.pointsEdgeLeft[i].y >= col) {
				col = track.pointsEdgeLeft[i].y;
				index = i;
			}
			else {
				return i;
			}
		}
	}
	return -1;
}
int Crossroad::WholeBlock(Tracking track) {// 输出最远一行的index
	int index = 0;
	width_counter = 0;
	for (int i = 0; i < track.widthBlock.size(); i++) {
		if (track.widthBlock[i].y == 319) {
			width_counter++;
			index = i;
		}else if (width_counter != 0) {
			if (width_counter > 40) {
				return index;
			}
			else
				width_counter = 0;
		}
	}
	return -1;
}