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
#include <cstdlib>
#include <cxcore.h>
#include <cvaux.h>
#include <highgui.h>
#include <iostream>
//#include "rrdb.h" // for statistics, testing
//#include <time.h> // For statistics, testing.
//#include <timeutils.h> // for statistics, testing.

/* Class for Silhouette Extraction. Should include everything needed for the extraction. */
class SilhouetteExtraction {
 public:
  SilhouetteExtraction();
  ~SilhouetteExtraction();
  SilhouetteExtraction(std::string input, int num_training_frames, bool isFile);
 
  //Returns the next frame from video or camera
  cv::Mat nextFrame();

  // Returns the input name (filename or cameraname).
  // Returns the string "UNSET" if not initialized
  std::string getInputName();

  // Initialize grabber, allocate space etc.
  void initialize(std::string clFileName);

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
  void performTrainingStep();

  /* Perform silhouette extraction for next image.
   * This involves copying data of the next image to GPU,
   * converting to color and luma, calculating/updating the distribution,
   * performing multiple morphological processes, extracting silhouette,
   * updating, etc. Need several helper-functions for this, I guess. */
  bool extractFromNextFrame();

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

 private:
  // no copy construction or assignment due to RRDB
  SilhouetteExtraction(const SilhouetteExtraction&);
  SilhouetteExtraction& operator=(const SilhouetteExtraction&);

  void convertNextFrame();
  void updateBgModelForFrame();
  void subtractBackgroundAndThreshold();
  void closeAndOpenImage();
  void labelConnectedComponents();  // For alg. 1
  void labelConnectedComponents2(); // For alg. 2
  void updateBG(); // Update mean and variance for BG regions in the end.
  void setupBuffersForExtraction();
  bool inputIsFile;
  bool hasInputSource;
  int debug;
  cv::Mat currentFrame; // currently grabbed frame
  
  cv::Mat outputFrame; // don't know output format yet, will have to change this TODO

  std::string inputFile; // Optional, if using a file as input
  std::string inputCam; // Optional, if using a camera as input
  std::string outFile; // Optional, if writing to an output-file

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

  /* Output test. TODO: remove*? */
  uchar* outTest;
  uchar* outTest2;
  uint32_t* uint32_image;
  int* int_image;

  /* Stuff needed for OpenCL */
  char *clFile;
  size_t kernels_size;
  std::vector<cl::Platform> cl_platforms;
  std::vector<cl::Device> cl_devices;
  cl::Program cl_program;
  //  cl_context_properties cl_properties;
  cl::Context cl_context;
  cl::Program::Sources cl_sources;
  cl::CommandQueue cl_cmdqueue;
  cl::Buffer cl_DistMap;
  cl::Buffer cl_rgb_frame; // used for one rgb frame at a time
  cl::Buffer cl_Mean_and_Variance;
  cl::Buffer cl_M2;
  cl::Buffer cl_M3;
  cl::Buffer cl_M4;
  cl::Buffer cl_Data;
  cl::Buffer cl_Model;
  cl::Buffer cl_uint_image;
  cl::Buffer cl_int_image;
  cl::Buffer cl_uint_image2; //
  cl::Buffer cl_uint_val;
  cl::Buffer cl_region_values;

  /* Used for closeAndOpen2. TODO: remove when tested */
  void closeAndOpenImage2();
  cl::Buffer cl_fgmask;
  cl::Buffer cl_fgmask2;
  cl::Kernel cl_kernel_dilate_2;
  cl::Kernel cl_kernel_erode_2;
  /* End closeAndOpen2 */

  /* Used for closeAndOpen3. TODO: remove, or keep if faster */
  void closeAndOpenImage3();
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

  cl::Kernel cl_kernel_convert_rgb_img;
  cl::Kernel cl_kernel_mean_variance_kurtosis;
  cl::Kernel cl_kernel_update_bg_model;
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
  cl::Kernel cl_kernel_update_bg;


  bool firstRun; // testing. TODO: remove?

  /* For performance-testing/statistics */
  //  RRDB<uint64_t> stats;
  /* RRDB<int64_t> trainingstep_stats; */
  /* RRDB<int64_t> bgsub_stats; */
  /* RRDB<int64_t> convert_stats; */
  /* RRDB<int64_t> update_stats; */
  /* RRDB<int64_t> closeopen_stats; */
  /* RRDB<int64_t> ccl_stats; */
  /* RRDB<int64_t> update_bg_stats; */
  /* RRDB<int64_t> total_stats; */
  /* FILE* logFile; */
};

#endif // SILHOUETTE_EXTRACTION_H
