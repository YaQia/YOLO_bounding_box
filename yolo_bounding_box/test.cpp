#include "box.h"
using namespace hitcrt;
/**
 * Author:姚益祁
 * Modified date:2020/12/06
 * Phone:18676436991
 * QQ:1210364094
 * Note: 如果有任何不懂的代码，可以直接通过注释的形式push上去，我会及时看到并修改
 **/
int main()
{
    // 记得修改路径，我这边是因为Detector相对路径用不了所以只能用绝对路径
    std::string names_file = "/home/ethan/Documents/rc2021/yolo_bounding_box/data/voc.names";
    std::string cfg_file = "/home/ethan/Documents/rc2021/yolo_bounding_box/data/yolov4-tiny-obj.cfg";
    std::string weights_file = "/home/ethan/Documents/rc2021/yolo_bounding_box/data/yolov4-tiny-obj_final.weights";
    std::string video_file = "/home/ethan/Documents/rc2021/yolo_bounding_box/data/1.flv";
    cv::VideoCapture video(video_file);
    std::vector<cv::Point2f> points[20];
    // 使用bbox_to_points的例子，容器数组的大小根据YOLO学习中names的多少来判断
    box bounding_box(names_file, cfg_file, weights_file);
    cv::Mat frame, dst;
    video >> frame;
    char key = '\0';
    while(!frame.empty() && key != 'q' && bounding_box.box_detect(frame))
    {
        bounding_box.print_info();
        bounding_box.bbox_to_points(bounding_box.get_bbox(), points);
        // bounding_box中有get_bbox()的接口
        cv::imshow("frame", frame);
        video >> frame;
        key = cv::waitKey(1);
        if(key == ' ')
        {
            cv::waitKey(0);
        }
    }

}