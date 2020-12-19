/**
 * Author:姚益祁
 * Modified date:2020/12/06
 * Phone:18676436991
 * QQ:1210364094
 * Note: 如果有任何不懂的代码，可以直接通过注释的形式push上去，我会及时看到并修改
 **/
#ifndef YOLO_BOUNDING_BOX_H
#define YOLO_BOUNDING_BOX_H

#define OPENCV 4.4.0
// 一定要定义这个OPENCV，否则在darknet中的接口将无法使用！
// 定义根据你的Opencv版本来，Opencv3和Opencv4均可使用本代码
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/core/version.hpp>
#include <opencv2/opencv.hpp>
#include "yolo_v2_class.hpp"        // darknet中的C++API
//#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSIO_MINOR)"" CVAUX_STR(CV_VERSION_REVISION)
// 实现自动识别opencv版本的方法，注意要导入opencv2/core/version.hpp头文件
//#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")
// 将opencv_world4.4.0.lib静态库加入（Windows平台）
// 如果你的windows中没有opencv_world.lib，则为很多其他的lib文件，实际上结果是等价的
// 实际上在CMakeLists.txt中写好了就没有必要在这里加入这些代码
namespace hitcrt
{
/**
 * @brief box类实现了绘制boundingbox和直接将得到的bbox_t转为points的操作 \n
 * 继承了Detector类，实现了对Detector的进一步封装，如果有任何bug需要完善，请QQ联系
 * QQ 1210364094
 * @author 姚益祁
 */
class box: public Detector{
public:
    box(std::string names, std::string cfg, std::string weights);
    int box_detect(cv::Mat& frame);
    void draw_boxes(cv::Mat& frame, std::vector<bbox_t>& result_vec);
    std::vector<bbox_t>& get_bbox(void){return bounded_boxes;}
    std::string get_curr_name(bbox_t& curr_box)
    {
        if(bounded_boxes.empty())
        {
            return "null";
        }
        return names[curr_box.obj_id];
    }
    void print_info(void);
    bool bbox_to_points(std::vector<bbox_t>& result_vec, std::vector<cv::Point2f> result_points[]);
private:
    std::string names_file;
    std::string cfg_file;
    std::string weights_file;
    std::vector<bbox_t> bounded_boxes;
    // 这个变量存储的是该帧所有框选区域的信息
    bbox_t* curr_box = nullptr;
    // 这个变量用于存储当前正在绘制的box
    std::vector<std::string> names;
    // 存储所有标签的名字
};
}

#endif // YOLO_BOUNDING_BOX_H