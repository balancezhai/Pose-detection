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

#include <stdio.h>
#include <stdint.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <thread>
extern "C"
{
#include <ivas/ivas_kernel.h>
#include <gst/ivas/gstinferencemeta.h>
}

enum
{
  LOG_LEVEL_ERROR,
  LOG_LEVEL_WARNING,
  LOG_LEVEL_INFO,
  LOG_LEVEL_DEBUG
};

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define LOG_MESSAGE(level, ...) {\
  do {\
    char *str; \
    if (level == LOG_LEVEL_ERROR)\
      str = (char*)"ERROR";\
    else if (level == LOG_LEVEL_WARNING)\
      str = (char*)"WARNING";\
    else if (level == LOG_LEVEL_INFO)\
      str = (char*)"INFO";\
    else if (level == LOG_LEVEL_DEBUG)\
      str = (char*)"DEBUG";\
    if (level <= log_level) {\
      printf("[%s %s:%d] %s: ",__FILENAME__, __func__, __LINE__, str);\
      printf(__VA_ARGS__);\
      printf("\n");\
    }\
  } while (0); \
}
static int log_level = LOG_LEVEL_WARNING;



#define MAX_CHANNELS 40 //40
//#define PROFILING 1
#define FRAME_SIZE(w,h) ((w)*(h)*3) // frame size for RGB

struct _roi {
    uint32_t y_cord;
    uint32_t x_cord;
    uint32_t height;
    uint32_t width;
    double   prob;
	GstInferencePrediction *prediction;
};

typedef struct _ivas_ms_roi {
    uint32_t nobj;
    struct _roi roi[MAX_CHANNELS];
} ivas_ms_roi;


static uint32_t xlnx_multiscaler_align(uint32_t stride_in, uint16_t AXIMMDataWidth) {
    uint32_t stride;
    uint16_t MMWidthBytes = AXIMMDataWidth / 8;

    stride = ((( stride_in ) + MMWidthBytes - 1) / MMWidthBytes) * MMWidthBytes;
    return stride;
}

using namespace std;
using namespace cv;

static int Crop_one_bgr(
    IVASKernel *handle,
    IVASFrame *input[MAX_NUM_OBJECT],
    const ivas_ms_roi& roi_data,
    const Mat& bgrImg,
    int ind
)
{
    IVASFrameProps out_props = {0, };
    out_props.width = 128;//80
    out_props.height = 224;//176
    out_props.fmt = IVAS_VFMT_BGR8;
    uint32_t size = FRAME_SIZE(out_props.width, out_props.height);

    // 看起来 app_priv 在这一步就已经被初始化成了 gstbuff
    // 不知道这个地方到底这么处理的
    IVASFrame *out_ivas_frame = ivas_alloc_buffer(handle, size, IVAS_FRAME_MEMORY, &out_props);
    if (!out_ivas_frame)
    {
        printf("ERROR: IVAS MS: failed to allocate frame memory");
        return 0;
    }
    else
    {
        {
            // app_priv 专门用来放buffer的指针
            // 内存类型必须要是：IVAS_FRAME_MEMORY , 不然 app_priv 会被初始化成NULL
            // 在 IVAS_FRAME_MEMORY 的条件下，调用下面的函数来分配 buffer
            // 需要在前面初始化 使用的是 kernel   IVASBufAllocCBFunc alloc_func; 来进行初始化
            
            // 分配buffer的时候可能调用了 gst_query_add_allocation_meta  相关函数
            // 由后续插件知道，使用buffer里面是可以提取到GstVideoMeta的数据的
            // 因此 图像大小这些数据本身也被藏在了这个buffer内部
            // 大概这才是为什么使用 新建一个frame 注册一个 buffer的原因

            // 如果这样的话，可能 app_priv 本身就存储了frame大部分数据，可能只是重新映射了地址

            // IVASBufAllocCBFunc 由于函数被隐藏了，不知道具体实现
        
            // 此时 app_priv应该已经被准备好了 

            // 每一帧的 app_priv 被放到 了 prediction->sub_buffer
            roi_data.roi[ind].prediction->sub_buffer = (GstBuffer*)out_ivas_frame->app_priv;
            {
                cv::Rect ROI(roi_data.roi[ind].x_cord, roi_data.roi[ind].y_cord,
                              roi_data.roi[ind].width, roi_data.roi[ind].height);

                cv::Mat subbgr = bgrImg(ROI);

                //info 是一个操作的临时区域？
                GstMapInfo info;

                // buffer这个类并没有直接把data放在一个可以访问的位置
                // map这个操作主要是通过 info 来间接的访问
                // 此处map为写操作之后，直接操作info即可
                gst_buffer_map((GstBuffer *)out_ivas_frame->app_priv, &info, GST_MAP_WRITE);
                
                //直接操作的是data 就是从 info中提取的
                char *indata = (char *)info.data;

                //本质上都是操作  buffer <--map--> info.data 的内存数据 
                //并且 这个 buffer 地址都被放到了 prediction->sub_buffer
                cv::Mat subbgrResize(out_props.height, out_props.width, CV_8UC3, indata);
                resize(subbgr, subbgrResize, subbgrResize.size());
                
                // 使用完后同样需要释放掉 info这个量
                gst_buffer_unmap((GstBuffer *)out_ivas_frame->app_priv, &info);
            }
            out_ivas_frame->app_priv = NULL;
        }
        // 同样是由插件来实现的，不知道发生了什么
        // 释放的时候 app_priv 指向的地址似乎是保留的 
        ivas_free_buffer(handle, out_ivas_frame);
        out_ivas_frame = NULL;
        return 0;
    }
}

static int Crop_range_bgr(
    IVASKernel *handle,
    IVASFrame *input[MAX_NUM_OBJECT],
    const ivas_ms_roi& roi_data,
    const Mat& bgrImg,
    int start, int stop
)
{
    cv::Mat bgrClone=bgrImg.clone();
    for (int i = start; i < stop; i++)
    {
        Crop_one_bgr(handle, input, roi_data, bgrClone, i);
    }
    return 0;
}

void
Thread(int numThread, int start, int stop, std::function<void(int, int)> func )
{
    int totalLoop = stop - start;
    int nloopPerT = totalLoop / numThread;
    int left = totalLoop % numThread;
    if (left > 0)
    {
        nloopPerT += 1;
        left = totalLoop - nloopPerT * (numThread - 1);
    }
    else 
    {
        left = nloopPerT;
    }
    vector<thread> pool;
    for (int i = 0; i < numThread; i++)
    {
        pool.emplace_back( func, start + nloopPerT * i,
                                     start + nloopPerT * i + ((i == numThread - 1) ? left : nloopPerT) );
    }
    for (int i = 0; i < numThread; i++)
    {
        pool[i].join();
    }
}

static int xlnx_multiscaler_descriptor_create (IVASKernel *handle,
    IVASFrame *input[MAX_NUM_OBJECT], IVASFrame *output[MAX_NUM_OBJECT],
    const ivas_ms_roi& roi_data)
{
    IVASFrameProps out_props = {0, };

    IVASFrame *in_ivas_frame = input[0];

    if (in_ivas_frame->props.fmt == IVAS_VFMT_BGR8)
    {
    LOG_MESSAGE(LOG_LEVEL_DEBUG, "Input frame is in BGR8 format\n");

    Mat bgrImg(input[0]->props.height, input[0]->props.width, CV_8UC3, (char *)in_ivas_frame->vaddr[0]);
    Crop_range_bgr(
        handle,
        input,
        roi_data,
        bgrImg,
        0, roi_data.nobj);
    }
    else
    {
        LOG_MESSAGE(LOG_LEVEL_WARNING, "Unsupported color format %d \n", in_ivas_frame->props.fmt);
        return 0;
    }
    return 0;
}


static int parse_rect(IVASKernel * handle, int start,
      IVASFrame * input[MAX_NUM_OBJECT], IVASFrame * output[MAX_NUM_OBJECT],
      ivas_ms_roi &roi_data
      )
{
    IVASFrame *inframe = input[0];
    GstInferenceMeta *infer_meta = ((GstInferenceMeta *)gst_buffer_get_meta((GstBuffer *)
                                                              inframe->app_priv,
                                                          gst_inference_meta_api_get_type()));
    if (infer_meta == NULL)
    {
        LOG_MESSAGE(LOG_LEVEL_INFO, "ivas meta data is not available for crop");
        return false;
    }

    GstInferencePrediction *root = infer_meta->prediction;

    roi_data.nobj = 0;
    /* Iterate through the immediate child predictions */
    GSList *collects = gst_inference_prediction_get_children(root);
    for ( GSList *child_predictions = collects; child_predictions;
         child_predictions = g_slist_next(child_predictions))
    {
        GstInferencePrediction *child = (GstInferencePrediction *)child_predictions->data;

        /* On each children, iterate through the different associated classes */
        for (GList *classes = child->classifications;
             classes; classes = g_list_next(classes))
        {
            GstInferenceClassification *classification = (GstInferenceClassification *)classes->data;
            if (roi_data.nobj < MAX_CHANNELS)
            {
                int ind = roi_data.nobj;
                roi_data.roi[ind].y_cord = (uint32_t)child->bbox.y + child->bbox.y % 2;
                roi_data.roi[ind].x_cord = (uint32_t)child->bbox.x;
                roi_data.roi[ind].height = (uint32_t)child->bbox.height - child->bbox.height % 2;
                roi_data.roi[ind].width = (uint32_t)child->bbox.width - child->bbox.width % 2;
                roi_data.roi[ind].prob = classification->class_prob;
                roi_data.roi[ind].prediction = child;
                roi_data.nobj++;
            }
        }
    }
    g_slist_free(collects);
    return 0;
}
extern "C"
{


int32_t xlnx_kernel_start (IVASKernel *handle, int start /*unused */,
        IVASFrame *input[MAX_NUM_OBJECT], IVASFrame *output[MAX_NUM_OBJECT])
{
    int ret;
    uint32_t value = 0;
    ivas_ms_roi roi_data;
    parse_rect(handle, start, input, output, roi_data);
   /* set descriptor */
    xlnx_multiscaler_descriptor_create (handle, input, output, roi_data);

    return 0;
}

int32_t xlnx_kernel_init (IVASKernel *handle)
{
    handle->is_multiprocess = 1;        
    return 0;
}

uint32_t xlnx_kernel_deinit (IVASKernel *handle)
{
    return 0;
}

int32_t xlnx_kernel_done(IVASKernel *handle)
{
    /* dummy */
    return 0;
}

}
