#include "box.h"

yolo::box::box(std::string names, std::string cfg, std::string weights): Detector(cfg, weights)
{
    if(names.empty() || cfg.empty() || weights.empty())
    {
        std::cerr << "文件目录不能为空！" << std::endl;
        return ;
    }
    this->names_file = names;
    this->cfg_file = cfg;
    this->weights_file = weights;
    // x_3d、y_3d、z_3d用不到，不用太在意
    std::ifstream ifs_names(names);
    char* buffer = new char[100];
    if(! ifs_names.is_open())
    {
        std::cerr << "文件读取失败！" << std::endl; 
    }
    while(!ifs_names.eof())
    {
        ifs_names.getline(buffer, 100);
        this->names.push_back(buffer);
    }
}
/** 
 * @brief 调用darknet接口获得bbox_t
 * @param frame 传入当前帧
 *
 * @return 返回值用于判断转换是否成功
 **/
int yolo::box::box_detect(cv::Mat& frame)
{
    this->bounded_boxes.clear();
    std::shared_ptr<image_t> dst_image = mat_to_image_resize(frame);
    this->bounded_boxes = detect_resized(*dst_image, frame.size().width, frame.size().height);
    if(!bounded_boxes.empty())
    {
        this->draw_boxes(frame, bounded_boxes);
    }
    else
    {
        return 0;
    }
    return 1;
}
/** 
 * @brief 绘制boxes
 * @param frame         绘制当前帧（也可以是图像或画布）
 * @param result_vec    传入需要绘制的bbox_t容器
 *
 * @return 无
 **/
void yolo::box::draw_boxes(cv::Mat& frame, std::vector<bbox_t>& result_vec)
{
    cv::Rect2d result;
    cv::Point center;
    int x, y;
    for(auto& i : result_vec)
    {
        this->curr_box = &i;
        result.x=i.x;
        result.y=i.y;
        result.width=i.w;
        result.height=i.h;
        center.x=int(result.x+result.width/2);
        center.y=int(result.y+result.height/2);
        // 这一长串的代码是在保证boundingbox以1.2倍识别区域大小框选
        x=int(center.x-result.width/2*1.2);
        y=int(center.y-result.height/2*1.2);
        if (x<=0)
            result.x=0;
        else
            result.x=x;
        if (y<=0)
            result.y=0;
        else
            result.y=y;
        if (x+1.2*i.w>=frame.cols)
            result.width=frame.cols-x;
        else
            result.width=1.2*i.w;
        if (y+1.2*i.h>=frame.rows)
            result.height=frame.rows-y;
        else
            result.height=1.2*i.h;
        cv::Scalar color = obj_id_to_color(i.obj_id);
        cv::rectangle(frame, cv::Rect2d(i.x, i.y, i.w, i.h), color, 5);
        std::string prob = std::to_string(i.prob);
        cv::putText(frame, this->names[i.obj_id], cv::Point2f(i.x, i.y + i.h), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 3);
        cv::putText(frame, prob, cv::Point2f(i.x, i.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 3);
    }
}

/** 
 * @brief boundingbox到points的转换
 * @param result_vec    bbox_t的容器
 * @param result_points points的容器数组，一个容器存储四个点
 *
 * @return 返回值用于判断转换是否成功
 **/
bool yolo::box::bbox_to_points(std::vector<bbox_t>& result_vec, std::vector<cv::Point2f> result_points[])
{
    if(result_vec.empty())
    {
        return false;
    }
    size_t iter = 0;            // iteration
    for(auto& i : result_vec)
    {
        result_points[iter].emplace_back(cv::Point2f(i.x, i.y));
        result_points[iter].emplace_back(cv::Point2f(i.x + i.w, i.y));
        result_points[iter].emplace_back(cv::Point2f(i.x + i.w, i.y + i.h));
        result_points[iter].emplace_back(cv::Point2f(i.x, i.y + i.h));
        iter++;
    }
    return true;
}
// 这个函数过于简单，所以不写注释了
void yolo::box::print_info()
{
    // 当当前帧没有识别到任何东西的时候，name将被赋值为"null"
    std::string name = get_curr_name(*(this->curr_box));
    if(name == "null")
    {
        std::cout << std::endl;
        return ;
    }
    std::cout << name << ":" << std::endl;
    std::cout << "points:" << "(" << curr_box->x << "," << curr_box->y << ")" << '\t'; 
    std::cout << "points:" << "(" << curr_box->x + curr_box->w << "," << curr_box->y << ")" << '\t'; 
    std::cout << "points:" << "(" << curr_box->x + curr_box->w << "," << curr_box->y + curr_box->h << ")" << '\t'; 
    std::cout << "points:" << "(" << curr_box->x << "," << curr_box->y + curr_box->h << ")" << '\n'; 
}