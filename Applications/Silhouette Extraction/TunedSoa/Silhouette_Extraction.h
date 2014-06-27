/* Silhouette Extraction using GPU and OpenCL
* Algorithm based on paper:
* "Robust Foreground Extraction Technique Using Gaussian Family Model and Multiple Thresholds"
* by Kim et. al.
*
* File:     Silhouette_Extraction.h
* Author:   Kristian Haga Karstensen <kristianhk@linux.com>
* Contents: Library for silhouette extraction in video. Header file.
*
* This software is released under the terms of the GNU Lesser
* General Public License Version 2.1 (LGPLv2.1). See file COPYING.LESSER
* for details.
*/
#ifndef SILHOUETTE_EXTRACTION_H
#define SILHOUETTE_EXTRACTION_H
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include "oclUtils.h"
#include <stdio.h>
#include <cxcore.h>
#include <cvaux.h>
#include <highgui.h>
#include <omp.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <numeric>

using namespace std;

/* Class for Silhouette Extraction. Should include everything needed for the extraction. */
class SilhouetteExtraction {
    public:
        SilhouetteExtraction();
        ~SilhouetteExtraction();
        SilhouetteExtraction(string input, int num_training_frames, bool isFile, int type, bool tiles);

        //Returns the next frame from video or camera
        cv::Mat nextFrame();

        // Returns the input name (filename or cameraname).
        // Returns the string "UNSET" if not initialized
        string getInputName();

        // Initialize grabber, allocate space etc.
        void initialize(string clFileName, int workgroupSize);
        int getMaxWorkGroupSize(string clFileName);

        // Called by the algorithm when a frame is finished processing.
        // Should maybe have some sort of mutex here - depends on how we implement the rest. TODO
        // What this should take as input also depends on how we will represent the silhouettes. TODO 2
        // Just write it as Mat now, will probably be changed.
        void putResult(cv::Mat& frameDone);

        /* Perform the training step:
        * 1. Read N training frames. For each frame, do:
        * 2. Send to GPU to convert to color and luma, and store in Data (in GPU-memory),
        *    don't write back
        * 3. Calculate the running mean, variance (std. deviation is done on the fly: sqrt(variance))
        * 4. Calculate running excess kurtosis on last training frame, 
        *    set DistMap for each pixel (distribution to use). */
        bool performTrainingStep(int workgroupSize);

        /* Perform silhouette extraction for next image.
        * This involves copying data of the next image to GPU,
        * converting to color and luma, calculating/updating the distribution,
        * performing multiple morphological processes, extracting silhouette,
        * updating, etc. Need several helper-functions for this, I guess. */
        bool extractFromNextFrame(int workgroupSize);

        /* Set debug level */
        void setDebug(int debug_lvl);

        /* Get debug level */
        int getDebug();

        /* Get curFrameNum. returns -1 if capture is 
        uninitialized (-1 set in constructor) */
        int getCurFrameNum();

        /* get N */
        int getN();

        /* Get frame count */
        int getFrameCount();

        /* TODO: for testing, remove this */
        void showOutput();

        void printStats();
        void avg(struct timeval start, struct timeval end, vector<float>& avg, int nin);
    
        string platformName;
    private:
        // no copy construction or assignment due to RRDB
        SilhouetteExtraction(const SilhouetteExtraction&);
        SilhouetteExtraction& operator=(const SilhouetteExtraction&);

        double convertRGBtoYH(int local_x, int local_y, int ts);
        double soaConvertRGBtoYH(int local_x, int local_y, int ts);
    
        double calculateKurtosis(int local_x, int local_y, int ts, int i);
        double soaCalculateKurtosis(int local_x, int local_y, int ts, int i);
    
        double updateBgModel(int local_x, int local_y);
        double soaUpdateBgModel(int local_x, int local_y);
    
        double subtractBgThresh(int local_x, int local_y);
        double soaSubtractBgThresh(int local_x, int local_y);
    
        double updateBackground(int local_x, int local_y);
        double soaUpdateBackground(int local_x, int local_y);
    
        void convertNextFrame(int workgroupSize);
        void updateBgModelForFrame(int workgroupSize);
        void subtractBackgroundAndThreshold(int workgroupSize);
        void closeAndOpenImage(int workgroupSize);
        void labelConnectedComponents(int workgroupSize);  // For alg. 1
        void labelConnectedComponents2(int workgroupSize); // For alg. 2
        void updateBG(int workgroupSize); // Update mean and variance for BG regions in the end.
        void setupBuffersForExtraction();
        bool inputIsFile;
        bool hasInputSource;
        int debug;
    
        cv::Mat currentFrame; // currently grabbed frame
        cv::Mat outputFrame; // don't know output format yet, will have to change this TODO

        string inputFile; // Optional, if using a file as input
        string inputCam; // Optional, if using a camera as input
        string outFile; // Optional, if writing to an output-file

        /* Video capture (using OpenCV to read a video file) */
        cv::VideoCapture cap;

        /* Thresholds and values for silhouette extraction */
        int width; // image width
        int height; // image height
        int N; // number of traning-frames
        int curFrameNum; // frame # for offset in data. Max N-1. Wraps around.
        int frameCount;
        int size; // (width*height)
        int videoFormat; // cv::Mat videoformat TODO: remove?
        int fps; // video fps TODO: remove?
        int type;
        bool tiling;

        /* Maps for choice of distribution for each pixel.
        * Chosen after calculating excess kurtosis in the training
        * step, and used throughout the program. */
        /* int* DistMap; */

        /* Buffers for data for N frames */
        /* int* Data; Not needed? OpenCL-runtime can take care of this itself. */

        /* Buffer for mean and variance */
        /* float* Mean_and_Variance; */

        /* Background model */
        /* float* Model; */

        /* Output test */
        uchar* outTest;
        uchar* outTest2;
        uint32_t* uint32_image;
        int* int_image;

        /* Stuff needed for OpenCL */
        char *clFile;
        size_t kernels_size;
        vector<cl::Platform> cl_platforms;
        vector<cl::Device> cl_devices;
        cl::Program cl_program;
        //  cl_context_properties cl_properties;
        cl::Context cl_context;
        cl::Program::Sources cl_sources;
        cl::CommandQueue cl_cmdqueue;
        cl::Buffer cl_DistMap;
        cl::Buffer cl_DistMap1;
        cl::Buffer cl_DistMap2;
        cl::Buffer cl_rgb_frame; // used for one rgb frame at a time
        cl::Buffer cl_Mean_and_Variance;
        cl::Buffer cl_Mean_and_Variance1;
        cl::Buffer cl_Mean_and_Variance2;
        cl::Buffer cl_Mean_and_Variance3;
        cl::Buffer cl_Mean_and_Variance4;
        cl::Buffer cl_M2;
        cl::Buffer cl_M21;
        cl::Buffer cl_M22;
        cl::Buffer cl_M3;
        cl::Buffer cl_M31;
        cl::Buffer cl_M32;
        cl::Buffer cl_M4;
        cl::Buffer cl_M41;
        cl::Buffer cl_M42;
        cl::Buffer cl_Data;    
        cl::Buffer cl_Data1;
        cl::Buffer cl_Data2;
        cl::Buffer cl_r;
        cl::Buffer cl_g;
        cl::Buffer cl_b;
        cl::Buffer cl_Model;
        cl::Buffer cl_Model1;
        cl::Buffer cl_Model2;
        cl::Buffer cl_uint_image;
        cl::Buffer cl_int_image;
        cl::Buffer cl_uint_image2; //
        cl::Buffer cl_uint_val;
        cl::Buffer cl_region_values;

        /* Used for closeAndOpen2. TODO: remove when tested */
        void closeAndOpenImage2(int workgroupSize);
        cl::Buffer cl_fgmask;
        cl::Buffer cl_fgmask2;
        cl::Kernel cl_kernel_dilate_2;
        cl::Kernel cl_kernel_erode_2;
        /* End closeAndOpen2 */

        /* Used for closeAndOpen3. TODO: remove, or keep if faster */
        void closeAndOpenImage3(int workgroupSize);
        cl::Kernel cl_kernel_closeandopen;
        // Reuse cl_fgmask and cl_fgmask2 from above. Move here if closeAndOpen2 is removed.
        /* End closeAndOpen3 */

        /* Used for thresholded output and dilate/erode input */
        cl::ImageFormat bin_img_format;
        cl::ImageFormat uint32_format;
        cl::Image2D image2d_fgmask;
        cl::Image2D image2d_fgmask2;
        cl::Image2D image2d_ccl;
        cl::Image2D image2d_ccl2;

        cl::Kernel cl_soa_kernel_convert_rgb_img;
        cl::Kernel cl_kernel_convert_rgb_img;
    
        cl::Kernel cl_soa_kernel_mean_variance_kurtosis;
        cl::Kernel cl_kernel_mean_variance_kurtosis;
    
        cl::Kernel cl_soa_kernel_update_bg_model;
        cl::Kernel cl_kernel_update_bg_model;

        cl::Kernel cl_soa_kernel_subtract_bg_and_threshold;
        cl::Kernel cl_kernel_subtract_bg_and_threshold;
    
        cl::Kernel cl_kernel_dilate;
        cl::Kernel cl_kernel_erode;
    
        cl::Kernel cl_kernel_ccl_init_labels;
        cl::Kernel cl_kernel_ccl_analysis;
        cl::Kernel cl_kernel_ccl_scanning;
        cl::Kernel cl_kernel_ccl_calculate_size;
    
        cl::Kernel cl_kernel_ccl2_1;
        cl::Kernel cl_kernel_ccl2_2;
        cl::Kernel cl_kernel_ccl2_3;
        cl::Kernel cl_kernel_ccl2_4;
    
        cl::Kernel cl_soa_kernel_update_bg;
        cl::Kernel cl_kernel_update_bg;

        bool firstRun; // testing. TODO: remove?
};

#endif // SILHOUETTE_EXTRACTION_H
