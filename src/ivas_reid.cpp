/*
 * Copyright 2021 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gst/ivas/gstinferencemeta.h>
#include <ivas/ivas_kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vitis/ai/nnpp/reid.hpp>
#include <vitis/ai/reid.hpp>
#include <vitis/ai/posedetect.hpp>
#include <vitis/ai/reidtracker.hpp>
#include "common.hpp"
#include <sstream>

#define MAX_REID 20
#define DEFAULT_REID_THRESHOLD 0.2
#define DEFAULT_REID_DEBUG     0
#define DEFAULT_MODEL_NAME     "personreid-res18_pt"
#define DEFAULT_MODEL_PATH     "/opt/xilinx/share/vitis_ai_library/models/kv260-aibox-reid"

using namespace std;

struct _Face {
  int last_frame_seen;
  int xctr;
  int yctr;
  int id;
  cv::Mat features;
};

typedef struct _kern_priv {
  uint32_t debug;
  double threshold;
  std::string modelpath;
  std::string modelname;
  // shared_ptr智能指针
  // 多个 shared_ptr 智能指针可以共同使用同一块堆内存
  // 用时需添加头文件<memory>
  // 　shared_ptr使用引用计数，每一个shared_ptr的拷贝都指向相同的内存。
  // 每使用他一次，内部的引用计数加1，每析构一次，内部的引用计数减1，
  // 减为0时，删除所指向的堆内存。
  // shared_ptr内部的引用计数是安全的，但是对象的读取需要加锁。


  // 从AI library 里面把 REID 和 REID TRACKER 拿出来
  // reid 应用
  //std::shared_ptr<vitis::ai::Reid> det;
  std::shared_ptr<vitis::ai::PoseDetect> det;
  // 这个类在 UG1354 1.4 没有放出
  //std::shared_ptr<vitis::ai::ReidTracker> tracker;
} ReidKernelPriv;

// ROI (region of interest)
// OPENCV中表示图像处理只会在该目标区域进行
struct _roi {
    uint32_t y_cord;
    uint32_t x_cord;
    uint32_t height;
    uint32_t width;
    double   prob;
	  GstInferencePrediction *prediction;
};

#define MAX_CHANNELS 40 //40

typedef struct _ivas_ms_roi {
    uint32_t nobj;
    struct _roi roi[MAX_CHANNELS];
    //define keypoints
    //struct _roi keypoint[MAX_KEYPOINTS];
} ivas_ms_roi;

static int parse_rect(IVASKernel * handle, int start,
      IVASFrame * input[MAX_NUM_OBJECT], IVASFrame * output[MAX_NUM_OBJECT],
      ivas_ms_roi &roi_data, GstInferencePrediction *root
      )
{
    // 把推理数据拿出来
    IVASFrame *inframe = input[0];
    GstInferenceMeta *infer_meta = ((GstInferenceMeta *)gst_buffer_get_meta((GstBuffer *)
                                                              inframe->app_priv,
                                                          gst_inference_meta_api_get_type()));
    if (infer_meta == NULL)
    {
        printf("No inference info for ReID.");
        return false;
    }

    // 预测结果
    
    //GstInferencePrediction *root = infer_meta->prediction;
    root = infer_meta->prediction;
    roi_data.nobj = 0;
    /* Iterate through the immediate child predictions */
    //迭代所有的预测框
    GSList *tmp = gst_inference_prediction_get_children(root);
    for (GSList *child_predictions = tmp;
         child_predictions;
         child_predictions = g_slist_next(child_predictions))
    {
        GstInferencePrediction *child = (GstInferencePrediction *)child_predictions->data;

        /* On each children, iterate through the different associated classes */
        // 遍历所有的分类结果
        for (GList *classes = child->classifications;
             classes; classes = g_list_next(classes))
        {
            GstInferenceClassification *classification = (GstInferenceClassification *)classes->data;
            if (roi_data.nobj < MAX_CHANNELS)
            {
                //把分类结果放到 roi_data 里面
                int ind = roi_data.nobj;
                struct _roi &roi = roi_data.roi[ind];
                roi.y_cord = (uint32_t)child->bbox.y + child->bbox.y % 2;
                roi.x_cord = (uint32_t)child->bbox.x;
                roi.height = (uint32_t)child->bbox.height - child->bbox.height % 2;
                roi.width = (uint32_t)child->bbox.width - child->bbox.width % 2;
                roi.prob = classification->class_prob;
                roi.prediction = child;
                roi_data.nobj++;

            }
        }
    }
    g_slist_free(tmp);
    return 0;
}

extern "C" {

// 初始化函数
int32_t xlnx_kernel_init(IVASKernel *handle) {
  // 拿到json文件
  json_t *jconfig = handle->kernel_config;
  json_t *val; /* kernel config from app */

  //多线程处理？
  handle->is_multiprocess = 1;
  //创建一个自定义存储数据的类
  ReidKernelPriv *kernel_priv =
      (ReidKernelPriv *)calloc(1, sizeof(ReidKernelPriv));
  if (!kernel_priv) {
    printf("Error: Unable to allocate reID kernel memory\n");
  }

  /* parse config */
  //解析json文件
  val = json_object_get(jconfig, "threshold");
  if (!val || !json_is_number(val))
    kernel_priv->threshold = DEFAULT_REID_THRESHOLD;
  else
    kernel_priv->threshold = json_number_value(val);

  val = json_object_get(jconfig, "debug");
  if (!val || !json_is_number(val))
    kernel_priv->debug = DEFAULT_REID_DEBUG;
  else
    kernel_priv->debug = json_number_value(val);

  val = json_object_get(jconfig, "model-name");
  if (!val || !json_is_string (val))
    kernel_priv->modelname = DEFAULT_MODEL_NAME;
  else
    kernel_priv->modelname = (char *) json_string_value (val);

  val = json_object_get(jconfig, "model-path");
  if (!val || !json_is_string (val))
    kernel_priv->modelpath = DEFAULT_MODEL_PATH;
  else
    kernel_priv->modelpath = (char *) json_string_value (val);

  //模型应该存在的完整路径
  std::string xmodelfile = kernel_priv->modelpath + "/" + kernel_priv->modelname + "/" + kernel_priv->modelname + ".xmodel";
  
  //使用这个完整路径来加载xmodel
  //kernel_priv->det = vitis::ai::Reid::create(xmodelfile);
  kernel_priv->det = vitis::ai::PoseDetect::create(xmodelfile);
  if (kernel_priv->det.get() == NULL) {
    printf("Error: Unable to create Reid runner with model %s.\n", xmodelfile.c_str());
  }
  // 这个部分其实是自己写的
  // 可能是后处理的库
  //kernel_priv->tracker = vitis::ai::ReidTracker::create();

  handle->kernel_priv = (void *)kernel_priv;
  return 0;
}

uint32_t xlnx_kernel_deinit(IVASKernel *handle) {
  ReidKernelPriv *kernel_priv = (ReidKernelPriv *)handle->kernel_priv;
  free(kernel_priv);
  return 0;
}

int32_t xlnx_kernel_start(IVASKernel *handle, int start /*unused */,
                          IVASFrame *input[MAX_NUM_OBJECT],
                          IVASFrame *output[MAX_NUM_OBJECT]) 
{
  IVASFrame *in_ivas_frame = input[0];
  ReidKernelPriv *kernel_priv = (ReidKernelPriv *)handle->kernel_priv;
  if ( !kernel_priv->det.get() ) { //|| !kernel_priv->tracker.get() 
    return 1;
  }
  // 帧数量
  // 此处 为 static 
  static int frame_num = 0;
  frame_num++;

  //std::vector<vitis::ai::ReidTracker::InputCharact> input_characts;
  /* get metadata from input */

  //把数据拿出来放到roi_data里面
  //感觉意义不大啊？
  ivas_ms_roi roi_data;
  GstInferencePrediction *root;
  parse_rect(handle, start, input, output, roi_data, root);

   //IVASFrame *inframe = input[0];
    GstInferenceMeta *infer_meta = ((GstInferenceMeta *)gst_buffer_get_meta((GstBuffer *)
                                                              in_ivas_frame->app_priv,
                                                          gst_inference_meta_api_get_type()));
  m__TIC__(getfeat);
  // 变量所有的roi目标
  for (uint32_t i = 0; i < roi_data.nobj; i++) 
  {
    // 感觉这写法很多余
    struct _roi& roi = roi_data.roi[i];
    {
      // roi 就是 roi_data[i]
      // 从input[0]里面拿出来的
      // 此处就是之前的 out_ivas_frame->app_priv
      GstBuffer *buffer = (GstBuffer *)roi.prediction->sub_buffer; /* resized crop image*/
      int xctr = roi.x_cord + roi.width / 2;
      int yctr = roi.y_cord + roi.height / 2;
      
      //同样是内存映射
      //info.data 映射到 buffer内部
      GstMapInfo info;
      gst_buffer_map(buffer, &info, GST_MAP_READ);

      // 很奇怪 frame 本身应该已经被释放掉了
      // IVASBufAllocCBFunc alloc_func; 
      // 内部应该是调用了 gst_query_add_allocation_meta
      // 所以此处而已拿到 vmeta 

      // vmeta的作用就是单纯的取得宽度 和 长度
      GstVideoMeta *vmeta = gst_buffer_get_video_meta(buffer);
      
      if (!vmeta) {
        printf("ERROR: IVAS REID: video meta not present in buffer");
      } else if (vmeta->width == 128 && vmeta->height == 224) { // 89*176
        char *indata = (char *)info.data;

        // indata 就是实际取得的图像
        cv::Mat image(vmeta->height, vmeta->width, CV_8UC3, indata);
        
        auto input_box =
            cv::Rect2f(roi.x_cord, roi.y_cord,
                       roi.width, roi.height);
        int cols = roi.width;
        int rows = roi.height;
        m__TIC__(reidrun);
        auto result = kernel_priv->det->run(image);
        m__TOC__(reidrun);
        m__TIC__(inputpush);
       // input_characts.emplace_back(feat, input_box, roi.prob, -1, i);
        m__TOC__(inputpush);
        if (kernel_priv->debug == 2) {
            printf("Tracker input: Frame %d: obj_ind %d, xmin %u, ymin %u, xmax %u, ymax %u, prob: %f\n",
                    frame_num, i, roi.x_cord, roi.y_cord,
                       roi.x_cord + roi.width,
                       roi.y_cord + roi.height, roi.prob);
        }

        //process pose detect
        //printf("xxxxxxxxxr\n");
          std::vector<cv::Point2f> pose14pt_arry = {result.pose14pt.right_shoulder, result.pose14pt.right_elbow, 
   result.pose14pt.right_wrist, result.pose14pt.left_shoulder, result.pose14pt.left_elbow, result.pose14pt.left_wrist, result.pose14pt.right_hip,
   result.pose14pt.right_knee, result.pose14pt.right_ankle, result.pose14pt.left_hip, result.pose14pt.left_knee, result.pose14pt.left_ankle,
   result.pose14pt.head, result.pose14pt.neck};
    //printf("tttttttt\n");
      int t= 0;
      for (auto & box:pose14pt_arry) {
        float xmin = box.x * cols + roi.x_cord ;
        float ymin = box.y * rows + roi.y_cord;
        float xmax = xmin + 1;
        float ymax = ymin + 1;
        float confidence = 888;

        BoundingBox bbox;
        GstInferencePrediction *predict;
        GstInferenceClassification *c = NULL;

        bbox.x = xmin;
        bbox.y = ymin;
        bbox.width = 1;
        bbox.height = 1;

        predict = gst_inference_prediction_new_full (&bbox);
        c = gst_inference_classification_new_full (-1, confidence,
        NULL, 0, NULL, NULL, NULL);
        gst_inference_prediction_append_classification (predict, c);
        // very important: append the child bbox to the main prediction, not the child of prediction
        gst_inference_prediction_append (infer_meta->prediction, predict);

      }
      } else {
        printf("ERROR: IVAS REID: Invalid resolution for reid (%u x %u)\n",
               vmeta->width, vmeta->height);
      }
      gst_buffer_unmap(buffer, &info);
    }
  }
  m__TOC__(getfeat);
  // if (input_characts.size() > 0)
  // {
  // std::vector<vitis::ai::ReidTracker::OutputCharact> track_results =
  //     std::vector<vitis::ai::ReidTracker::OutputCharact>(
  //         kernel_priv->tracker->track(frame_num, input_characts, true, true));
  // if (kernel_priv->debug) {
  //     printf("Tracker result: \n");
  // }
  // int i = 0;
  // for (auto &r : track_results) {
  //   auto box = get<1>(r);
  //   gint tmpx = box.x, tmpy = box.y;
  //   guint tmpw = box.width, tmph = box.height;
  //   uint64_t gid = get<0>(r);

  //   if (kernel_priv->debug) {
  //     printf("Frame %d: %" PRIu64 ", xmin %d, ymin %d, w %u, h %u\n",
  //        frame_num, gid,
  //        tmpx, tmpy,
  //        tmpw, tmph);
  //   }

  //   struct _roi& roi = roi_data.roi[i];
  //   roi_data.roi[i].prediction->bbox.x = tmpx;
  //   roi_data.roi[i].prediction->bbox.y = tmpy;
  //   roi_data.roi[i].prediction->bbox.width = tmpw;
  //   roi_data.roi[i].prediction->bbox.height = tmph;
  //   roi_data.roi[i].prediction->reserved_1 = (void*)gid;
  //   roi_data.roi[i].prediction->reserved_2 = (void*)1;

  //   i++;
  // }

  // for (; i < roi_data.nobj; i++)
  // {
  //   roi_data.roi[i].prediction->reserved_2 = (void*)-1;
  // }
  // }
  return 0;
}

int32_t xlnx_kernel_done(IVASKernel *handle) {
  /* dummy */
  return 0;
}
}
