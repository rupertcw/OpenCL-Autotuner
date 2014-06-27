/* Silhouette Extraction using GPU and OpenCL
* Algorithm based on paper:
* "Robust Foreground Extraction Technique Using Gaussian Family Model and Multiple Thresholds"
* by Kim et. al.
*
* File:     Silhouette_Extraction.cpp
* Author:   Kristian Haga Karstensen <kristianhk@linux.om>
* Contents: Library for silhouette extraction in video.
*
* This software is released under the terms of the GNU Lesser
* General Public License Version 2.1 (LGPLv2.1). See file COPYING.LESSER
* for details.
*/
#include "Silhouette_Extraction.h"

using namespace std;

vector<float> avg1;
vector<float> avg2;
vector<float> avg3;
vector<float> avg4;
vector<float> avg5;
vector<float> avg6;

typedef struct {
    uchar* red;
    uchar* green;
    uchar* blue;
} soa_rgb;

SilhouetteExtraction::SilhouetteExtraction(string input, int num_training_frames, bool isFile, int typ, bool tiles)
/* Needed for RRDB (statistics) */
// trainingstep_stats(10000),
// bgsub_stats(10000),
// convert_stats(10000),
// update_stats(10000),
// closeopen_stats(10000),
// ccl_stats(10000),
// update_bg_stats(10000),
// total_stats(10000)
{
    N = num_training_frames;
    curFrameNum = -1;
    frameCount = -1;
    
    /* If the input is a file, assign inName to inputFile.
    Otherwise we assume it is a camera name/identifier. */
    if(isFile) {
        inputFile = input;
    }
    else {
        inputCam = input;
    }
    inputIsFile = isFile;
    hasInputSource = true;

    debug = 0; // Start with debug level 0
    kernels_size = -1;
    clFile = NULL;
    // logFile = fopen("sil_extract.log", "a");
    
    type = typ;
    tiling = tiles;
}

SilhouetteExtraction::SilhouetteExtraction()//:
/* Needed for RRDB (statistics) */
// trainingstep_stats(10000),
// bgsub_stats(10000),
// convert_stats(10000),
// update_stats(10000),
// closeopen_stats(10000),
// ccl_stats(10000),
// update_bg_stats(10000),
// total_stats(10000)
{}

SilhouetteExtraction::~SilhouetteExtraction()
{
//    free(Data);
//    free(DistMap);
// TODO: more cleanup
// fclose(logFile);
}

//Returns the next frame from video or camera (as cv::Mat)
cv::Mat SilhouetteExtraction::nextFrame()
{
    if(!hasInputSource) {
        if(debug >= 1) {
            printf("--- WARNING --- No input source defined, returning empty cv::Mat from nextFrame()\n");
        }
        currentFrame.data = NULL;
        return cv::Mat();
    }

    if(inputIsFile) {
        if(cap.grab()) {
            curFrameNum++;
            frameCount++;
            if(curFrameNum == N)
                curFrameNum = 0;

            cap.retrieve(currentFrame,0);

            return currentFrame;
        }
        else {
            if(debug >= 1) {
                printf("---INFO --- No more frames in file. Returning empty cv::Mat from nextFrame().\n");
            }
            return cv::Mat();
        }
    }
    else {
    // not implemented. TODO
        return cv::Mat();
    }
}

// Returns the input name (filename or cameraname).
// Returns the string "UNSET" if not initialized
string SilhouetteExtraction::getInputName() {
    if(!hasInputSource) {
        return string("UNSET");
    }
    else {
        if(inputIsFile)
            return inputFile;
        else
            return inputCam;
    }
}

// Get max workgroup size
int SilhouetteExtraction::getMaxWorkGroupSize(string clFileName){
    int maxWorkgroupSize = 0;

    clFile = oclLoadProgSource(clFileName.c_str(), "", &kernels_size);

    if (clFile == NULL) {
        if(debug >= 0) {
            printf("--- ERROR --- Couldn't open .cl-file %s! Exiting.\n",clFileName.c_str());
        }
        exit(1);
    }

    // clFile = "__kernel void foo(global float *data){data[0] = 0.f;}";
    //cout << kernels_size << endl;

    /* Initialize device, program, context, kernels */
    try {
        cl::Platform::get(&cl_platforms);
        
        if(cl_platforms.size() == 0) {
            if(debug >= 0)
                printf("--- ERROR --- OpenCL platform-size is 0. Exiting.\n");
            exit(1);
        }
        
        cl_context_properties cl_properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(cl_platforms[1])(), 0};
        
        cl_context = cl::Context(CL_DEVICE_TYPE_GPU, cl_properties);
        
        cl_devices = cl_context.getInfo<CL_CONTEXT_DEVICES>();
        //cout << "Using " << cl_devices.size() << " devices " << endl;
        
        //////////////////////
        vector<cl::Platform> all_platforms;
        cl::Platform::get(&all_platforms);
        
        if(all_platforms.size()==0){
            cout << " No platforms found. Check OpenCL installation!" << endl;
            exit(1);
        }
        
        cl::Platform default_platform = all_platforms[1];
        platformName = default_platform.getInfo<CL_PLATFORM_NAME>();
        
        cout << "--- MAIN --- Using platform: " << platformName << endl;
        
        //////////////////////
        
        default_platform.getDevices(CL_DEVICE_TYPE_GPU, &cl_devices);
        if(cl_devices.size() == 0)
        {
            cout << " No devices found. Check OpenCL installation!" << endl;
            exit(1);
        }
        
        cl::Device default_device = cl_devices[0];
        
        cout << "--- MAIN --- Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << endl;
        //////////////////////
        
        cl_sources = cl::Program::Sources(1, make_pair(clFile, kernels_size));
        
        cl_program = cl::Program(cl_context, cl_sources);
        
        cl_program.build(cl_devices);
        
        cl_cmdqueue = cl::CommandQueue(cl_context, cl_devices[1]);
        
        // WG Size Tests
        
        cl_kernel_convert_rgb_img = cl::Kernel(cl_program, "convert_rgb_to_h_and_y");
        maxWorkgroupSize = (int)cl_kernel_convert_rgb_img.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(default_device);
        
        cout << "--- MAIN --- Max Workgroup Size: " << maxWorkgroupSize << endl;

//        cl_kernel_mean_variance_kurtosis = cl::Kernel(cl_program, "calc_mean_variance_kurtosis");
//        maxWorkgroupSize = (int)cl_kernel_mean_variance_kurtosis.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(default_device);
//
//        cout << "WG Size2: " << maxWorkgroupSize << endl;
//        
//        cl_kernel_update_bg_model = cl::Kernel(cl_program, "update_bg_model");
//        maxWorkgroupSize = (int)cl_kernel_update_bg_model.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(default_device);
//        
//        cout << "WG Size3: " << maxWorkgroupSize << endl;
//        
//        cl_kernel_subtract_bg_and_threshold = cl::Kernel(cl_program, "subtract_and_threshold");
//        maxWorkgroupSize = (int)cl_kernel_subtract_bg_and_threshold.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(default_device);
//        
//        cout << "WG Size4: " << maxWorkgroupSize << endl;
//        
//        cl_kernel_dilate = cl::Kernel(cl_program, "dilate_image");
//        maxWorkgroupSize = (int)cl_kernel_dilate.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(default_device);
//        
//        cout << "WG Size5: " << maxWorkgroupSize << endl;
//        
//        cl_kernel_erode = cl::Kernel(cl_program, "erode_image");
//        maxWorkgroupSize = (int)cl_kernel_erode.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(default_device);
//        
//        cout << "WG Size6: " << maxWorkgroupSize << endl;

    }
    catch (cl::Error cl_err) {
        printf("--- ERROR --- getMaxWGS --- %s (%s)\n", cl_err.what(), oclErrorString(cl_err.err()));

        if(cl_err.err() == CL_BUILD_PROGRAM_FAILURE) {
            cl::STRING_CLASS buildLog;
            cl_program.getBuildInfo(cl_devices[1], CL_PROGRAM_BUILD_LOG, &buildLog);
            cout << buildLog << endl;
        }
        exit(1);
    }

    return maxWorkgroupSize;
}

// Initialize grabber, allocate space etc.
void SilhouetteExtraction::initialize(string clFileName, int workgroupSize) {
    if(!hasInputSource) {
        if(debug >= 0) {
            printf("--- ERROR --- No input source defined. Exiting.\n");
            exit(1);
        }
    }

    if(inputIsFile) {
        cap = cv::VideoCapture(inputFile);
        curFrameNum = -1;
        frameCount = 0;

        // Check that the file can be opened.
        if(!cap.isOpened()) {
            if(debug >= 0) {
                printf("--- ERROR --- Video file \"%s\" could not be opened. Exiting.\n",inputFile.c_str());
                exit(1);
            }
        }

        // Get video properties from cap
        width = (int) cap.get(CV_CAP_PROP_FRAME_WIDTH);
        height = (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT);
        
        cout << "starting width: " << width << endl;
        while (width % workgroupSize != 0) {
            width++;
        }
        
        cout << "new width: " << width << endl;

        cout << "starting height: " << height << endl;
        while (height % workgroupSize != 0) {
            height++;
        }
        
        cout << "new height: " << height << endl;
        
        videoFormat = (int) cap.get(CV_CAP_PROP_FORMAT);
        fps = (int) cap.get(CV_CAP_PROP_FPS);
        size = width*height;
        currentFrame = cv::Mat(height, width, videoFormat);
        // Data = (int*) calloc((size*N)*2,sizeof(int)); // don't need this. OpenCL-runtime will allocate what it needs
        // DistMap = (int*) calloc(size*2,sizeof(int));
        // Mean_and_Variance = (float*) calloc(size,sizeof(float));
        // Model = (float*) calloc(size*2,sizeof(float));
        outTest = (uchar*) calloc(size, sizeof(uchar)); // TODO: remove?
        outTest2 = (uchar*) calloc(size, sizeof(uchar)); // TODO: remove?
        uint32_image = (uint32_t*) calloc(size, sizeof(uint32_t));
        int_image = (int*) calloc(size, sizeof(int));
    }
    // Else input is cam/stream
    else {
    // TODO: implement this when needed
    }

    /* Setup OpenCL stuff */
    clFile = oclLoadProgSource(clFileName.c_str(), "", &kernels_size);

    if(clFile == NULL) {
        if(debug >= 0)
            printf("--- ERROR --- Couldn't open .cl-file %s! Exiting.\n",clFileName.c_str());
        exit(1);
    }

    // clFile = "__kernel void foo(global float *data){data[0] = 0.f;}";
    //cout << kernels_size << endl;
    
    /* Initialize device, program, context, kernels */
    try {
        cl::Platform::get(&cl_platforms);

        if(cl_platforms.size() == 0) {
            if(debug >= 0)
                printf("--- ERROR --- OpenCL platform-size is 0. Exiting.\n");
            exit(1);
        }

        cl_context_properties cl_properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(cl_platforms[1])(), 0};

        cl_context = cl::Context(CL_DEVICE_TYPE_GPU, cl_properties);

        cl_devices = cl_context.getInfo<CL_CONTEXT_DEVICES>();
        //cout << "Using " << cl_devices.size() << " devices " << endl;

        //////////////////////
        vector<cl::Platform> all_platforms;
        cl::Platform::get(&all_platforms);
        
        if(all_platforms.size()==0){
            cout << " No platforms found. Check OpenCL installation!" << endl;
            exit(1);
        }
        
        cl::Platform default_platform = all_platforms[1];
//        cout << "--- MAIN --- Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>() << endl;

        default_platform.getDevices(CL_DEVICE_TYPE_GPU, &cl_devices);
        if(cl_devices.size() == 0){
            cout << " No devices found. Check OpenCL installation!\n";
            exit(1);
        }

        cl::Device default_device = cl_devices[0];
//        cout<< "--- MAIN --- Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

        cl_sources = cl::Program::Sources(1, make_pair(clFile, kernels_size));

        cl_program = cl::Program(cl_context, cl_sources);

        cl_program.build(cl_devices);

        cl_cmdqueue = cl::CommandQueue(cl_context, cl_devices[1]);

        /* Setup memory for usage throughout the rest of the video */
        setupBuffersForExtraction();

    }
    catch (cl::Error cl_err) {
        printf("--- ERROR --- Initialize --- %s (%s)\n", cl_err.what(), oclErrorString(cl_err.err()));

        if(cl_err.err() == CL_BUILD_PROGRAM_FAILURE){
            cl::STRING_CLASS buildLog;
            cl_program.getBuildInfo(cl_devices[1], CL_PROGRAM_BUILD_LOG, &buildLog);
            cout << buildLog << endl;
        }
        exit(1);
    }
}

// Called by the algorithm when a frame is finished processing.
// Should maybe have some sort of mutex here - depends on how we implement the rest. TODO
// What this should take as input also depends on how we will represent the silhouettes. TODO 2
// Just write it as Mat now, may have to be changed.
void SilhouetteExtraction::putResult(cv::Mat& frameDone) {
// Does nothing atm. Not sure if it should. TODO: remove?
}

double SilhouetteExtraction::soaConvertRGBtoYH(int local_x, int local_y, int ts)
{
    double start1 = omp_get_wtime();
    
    cl::Event cl_event;
    
    /* Allocate buffer for in-data (RGB frame data). */
    soa_rgb frame;
    frame.red   = (uchar*)malloc(sizeof(uchar)*size);
    frame.green = (uchar*)malloc(sizeof(uchar)*size);
    frame.blue  = (uchar*)malloc(sizeof(uchar)*size);
    
    for (int i=0; i<height; i++){
        for (int j=0; j<width; j++){
            frame.blue[i*width + j]  = currentFrame.at<cv::Vec3b>(i, j)[0];
            frame.green[i*width + j] = currentFrame.at<cv::Vec3b>(i, j)[1];
            frame.red[i*width + j]   = currentFrame.at<cv::Vec3b>(i, j)[2];
        }
    }
    
    cl_r = cl::Buffer(cl_context, CL_MEM_READ_ONLY, size*sizeof(uchar));
    cl_g = cl::Buffer(cl_context, CL_MEM_READ_ONLY, size*sizeof(uchar));
    cl_b = cl::Buffer(cl_context, CL_MEM_READ_ONLY, size*sizeof(uchar));
    
    cl_cmdqueue.enqueueWriteBuffer(cl_r, CL_TRUE, 0, size*sizeof(uchar), frame.red, NULL, NULL);
    cl_cmdqueue.enqueueWriteBuffer(cl_g, CL_TRUE, 0, size*sizeof(uchar), frame.green, NULL, NULL);
    cl_cmdqueue.enqueueWriteBuffer(cl_b, CL_TRUE, 0, size*sizeof(uchar), frame.blue, NULL, NULL);
    
    /* Set args for kernel */
    cl_soa_kernel_convert_rgb_img.setArg(0, cl_r);
    cl_soa_kernel_convert_rgb_img.setArg(1, cl_g);
    cl_soa_kernel_convert_rgb_img.setArg(2, cl_b);
    cl_soa_kernel_convert_rgb_img.setArg(6, 0);
    cl_soa_kernel_convert_rgb_img.setArg(7, ts);
    
    double start = omp_get_wtime();
    
    cl_cmdqueue.enqueueNDRangeKernel(cl_soa_kernel_convert_rgb_img,
                                     cl::NullRange, cl::NDRange(size),
                                     cl::NDRange(local_x*local_y), NULL, &cl_event);
    cl_event.wait();    // remove? for speed
    
    double end = omp_get_wtime();
    cout << "time for soa convertRGB kernel: " << end-start << endl;
    
    //            int test1[size], test2[size];
    //
    //            cl_cmdqueue.enqueueReadBuffer(cl_Data1, CL_TRUE, 0, size*sizeof(int), test1, NULL, NULL);
    //            cl_cmdqueue.enqueueReadBuffer(cl_Data2, CL_TRUE, 0, size*sizeof(int), test2, NULL, NULL);
    //
    //            for (int i=0; i<5; i++) {
    //                cout << "Y: " << test1[i] << endl;
    //                cout << "H: " << test2[i] << endl;
    //            }
    
    double end1 = omp_get_wtime();
    return end1-start1;

}

double SilhouetteExtraction::convertRGBtoYH(int local_x, int local_y, int ts)
{
    double start1 = omp_get_wtime();
    cl::Event cl_event;

    cl_rgb_frame = cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (3*size)*currentFrame.elemSize1(), (uchar*) currentFrame.data);

    cl_kernel_convert_rgb_img.setArg(0, cl_rgb_frame);
    cl_kernel_convert_rgb_img.setArg(3, 0);
    cl_kernel_convert_rgb_img.setArg(4, ts);

    double start = omp_get_wtime();

    cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_convert_rgb_img,
                                     cl::NullRange, cl::NDRange(width, height),
                                     cl::NDRange(local_x, local_y), NULL, &cl_event);
    cl_event.wait();    // remove? for speed

    double end = omp_get_wtime();
    cout << "time for def convertRGB kernel: " << end-start << endl;
    
    double end1 = omp_get_wtime();
    return end1-start1;
}

double SilhouetteExtraction::soaCalculateKurtosis(int local_x, int local_y, int ts, int i)
{
    double start1 = omp_get_wtime();
    
    cl::Event cl_event;
    
    /* Buffer for distribution-maps that say what type of GGF-distribution to use for each
     pixel (1 = Laplace, 2 = Gaussian) */
    cl_kernel_mean_variance_kurtosis.setArg(0, N);
    cl_kernel_mean_variance_kurtosis.setArg(1, size);
    cl_kernel_mean_variance_kurtosis.setArg(2, i); // count (frame #)
    
    double start = omp_get_wtime();
    cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_mean_variance_kurtosis,
                                     cl::NullRange, cl::NDRange(size),
                                     cl::NDRange(local_x*local_y), NULL, &cl_event);
    cl_event.wait();    // remove? for speed
    
    double end = omp_get_wtime();
    cout << "time for soa kurtosis kernel: " << end-start << endl;
    
    double end1 = omp_get_wtime();
    return end1-start1;
}

double SilhouetteExtraction::calculateKurtosis(int local_x, int local_y, int ts, int i)
{
    double start1 = omp_get_wtime();

    cl::Event cl_event;
    
    /* Buffer for distribution-maps that say what type of GGF-distribution to use for each
       pixel (1 = Laplace, 2 = Gaussian) */
    cl_kernel_mean_variance_kurtosis.setArg(0, N);
    cl_kernel_mean_variance_kurtosis.setArg(1, size);
    cl_kernel_mean_variance_kurtosis.setArg(2, i); // count (frame #)

    double start = omp_get_wtime();

    cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_mean_variance_kurtosis,
                                     cl::NullRange, cl::NDRange(size),
                                     cl::NDRange(local_x*local_y), NULL, &cl_event);
    cl_event.wait();    // remove? for speed
    
    double end = omp_get_wtime();
    cout << "time for def kurtosis kernel: " << end-start << endl;
    
    double end1 = omp_get_wtime();
    return end1-start1;
}

/* Perform the training step (for N frames).
* 1. Read frame.
* 2. Send to GPU to convert to color and luma, and store in GPU-mem.
* 3. Calculate running mean, running variance and what's needed for running excess kurtosis
* 4. In the last training-frame, calculate excess kurtosis and choose distribution for each pixel.
*    Store this in DistMap (in GPU-memory).
*/
bool SilhouetteExtraction::performTrainingStep(int workgroupSize) {
    
    cout << "--- SILHOUETTE EXTRACTION --- Perform Training Step" << endl;
    
    int local_x, local_y, ts = 8;
    double time1, time2, time3, time4;
    
    if (type == 0) {
        local_x = workgroupSize;
        local_y = 1;
    }
    else if (type == 1){
        local_x = 1;
        local_y = workgroupSize;
    } else {
        local_x = workgroupSize;
        local_y = workgroupSize;
    }

    double start1 = omp_get_wtime();
    
    try {
        /* In the training step we read N frames */
        for (int i=0;i<N;i++) {
            cv::Mat tmpFrame = nextFrame();
            if (tmpFrame.data == NULL) {
                printf("--- ERROR --- performTrainingStep(): Got empty frame. Exiting.\n");
                exit(1);
            }
            
            time1 = soaConvertRGBtoYH(local_x, local_y, ts);
            time2 = convertRGBtoYH(local_x, local_y, ts);
            
            time3 = soaCalculateKurtosis(local_x, local_y, ts, i);
            time4 = calculateKurtosis(local_x, local_y, ts, i);
        }

    }
    catch (cl::Error cl_err){
        printf("--- ERROR --- PerfTrainStep --- %s (%s)\n", cl_err.what(), oclErrorString(cl_err.err()));

        if(cl_err.err() == CL_INVALID_WORK_GROUP_SIZE) {
            return false;
        }
        else if(cl_err.err() == CL_BUILD_PROGRAM_FAILURE) {
            cl::STRING_CLASS buildLog;
            cl_program.getBuildInfo(cl_devices[1], CL_PROGRAM_BUILD_LOG, &buildLog);
            cout << buildLog << endl;
            exit(1);
        }
    }
    
    double end1 = omp_get_wtime();
    cout << "time for perfTrainingStep: " << end1-start1 << endl;
    
    return true;
}

/* Perform silhouette extraction for next image.
 This involves copying data of the next image to GPU,
 converting to color and luma, calculating/updating the distribution,
 performing multiple morphological processes, extracting silhouette,
 updating, etc. Need several helper-functions for this. */
bool SilhouetteExtraction::extractFromNextFrame(int workgroupSize)
{
    cv::Mat tmpFrame = nextFrame();
    
    if(tmpFrame.data == NULL)
        return false;

    /* Grab and convert a frame (color space conversion) */
    convertNextFrame(workgroupSize);

    // Update the background model from new frame data
    updateBgModelForFrame(workgroupSize);

    // Extract silhouettes.
    subtractBackgroundAndThreshold(workgroupSize);

    closeAndOpenImage(workgroupSize); // two kernels, image2d
    //closeAndOpenImage2(); // two kernels, global mem
    //closeAndOpenImage3(); // one kernel, global mem, local mem

    // Note: not drop-in replaceable functions - will need
    // to look into the different steps and change output/input
    // accordingly. This goes for all functions where there
    // are multiple choices of kernels/functions.

    //labelConnectedComponents2(workgroupSize);
    labelConnectedComponents(workgroupSize);

    updateBG(workgroupSize);

    return true;
}

/* Convert RGB image to Y (luma) and H (color)*/
void SilhouetteExtraction::convertNextFrame(int workgroupSize)
{
    cout << "--- SILHOUETTE EXTRACTION --- Convert Next Frame" << endl;
    cl::Event cl_event;
    
    int local_x, local_y, ts = 8;
    double time1, time2;
    
    if (type == 0) {
        local_x = workgroupSize;
        local_y = 1;
    }
    else if (type == 1){
        local_x = 1;
        local_y = workgroupSize;
    }
    else{
        local_x = workgroupSize;
        local_y = workgroupSize;
    }

    try {
        time1 = soaConvertRGBtoYH(local_x, local_y, ts);
        time2 = convertRGBtoYH(local_x, local_y, ts);
    }
    catch (cl::Error cl_err) {
        printf("--- ERROR --- convNextFrame --- %s (%s)\n", cl_err.what(), oclErrorString(cl_err.err()));

        if(cl_err.err() == CL_BUILD_PROGRAM_FAILURE) {
            cl::STRING_CLASS buildLog;
            cl_program.getBuildInfo(cl_devices[1], CL_PROGRAM_BUILD_LOG, &buildLog);
            cout << buildLog << endl;
        }
        exit(1);
    }

    if(debug >= 2) {
        printf("--- INFO --- extractFromNextFrame() - Frame %d: Converted color space.\n",frameCount);
    }
}

double SilhouetteExtraction::soaUpdateBgModel(int local_x, int local_y)
{
    double start1 = omp_get_wtime();
    
    cl::Event cl_event;
    
    double start = omp_get_wtime();
    
    cl_cmdqueue.enqueueNDRangeKernel(cl_soa_kernel_update_bg_model,
                                     cl::NullRange, cl::NDRange(width, height),
                                     cl::NDRange(local_x, local_y), NULL, &cl_event);
    cl_event.wait();    // remove? for speed
    
    double end = omp_get_wtime();
    cout << "time for soa updateBG kernel: " << end-start << endl;
    
    double end1 = omp_get_wtime();
    return end1-start1;
}

double SilhouetteExtraction::updateBgModel(int local_x, int local_y)
{
    double start1 = omp_get_wtime();
    
    cl::Event cl_event;
    
    double start = omp_get_wtime();
    
    cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_update_bg_model,
                                     cl::NullRange, cl::NDRange(width, height),
                                     cl::NDRange(local_x, local_y), NULL, &cl_event);
    cl_event.wait();    // remove? for speed
    
    double end = omp_get_wtime();
    cout << "time for def updateBG kernel: " << end-start << endl;
    
    double end1 = omp_get_wtime();
    return end1-start1;
}

void SilhouetteExtraction::updateBgModelForFrame(int workgroupSize)
{
    cout << "--- SILHOUETTE EXTRACTION --- Update Bg Model For Frame" << endl;
    
    int local_x;
    int local_y;
    double time1, time2;
    
    if (type == 0) {
        local_x = workgroupSize;
        local_y = 1;
    }
    else if (type == 1){
        local_x = 1;
        local_y = workgroupSize;
    }
    else{
        local_x = workgroupSize;
        local_y = workgroupSize;
    }

    /* Update background model for the current frame */
    try {
        time1 = soaUpdateBgModel(local_x, local_y);
        time2 = updateBgModel(local_x, local_y);
    }
    catch (cl::Error cl_err) {
        printf("--- ERROR --- updateBgModelForFrame --- %s (%s)\n",cl_err.what(),oclErrorString(cl_err.err()));

        if(cl_err.err() == CL_BUILD_PROGRAM_FAILURE) {
            cl::STRING_CLASS buildLog;
            cl_program.getBuildInfo(cl_devices[1], CL_PROGRAM_BUILD_LOG, &buildLog);
            cout << buildLog << endl;
        }
        exit(1);
    }

    if(debug >= 2) {
        printf("--- INFO --- updateBgModelForFrame() - Frame %d: Updated BG-model.\n",frameCount);
    }
}

double SilhouetteExtraction::soaSubtractBgThresh(int local_x, int local_y)
{
    double start1 = omp_get_wtime();
    
    cl::Event cl_event;
    
//    cl_uint_image = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(uint8_t)));
    cl_soa_kernel_subtract_bg_and_threshold.setArg(8, image2d_fgmask);
    cl_soa_kernel_subtract_bg_and_threshold.setArg(9, width);
    cl_soa_kernel_subtract_bg_and_threshold.setArg(10, cl_region_values);
    
    /* For closeAndOpenImage2 -  TODO: remove when done */
    cl_soa_kernel_subtract_bg_and_threshold.setArg(11, cl_fgmask);
    /* End closeAndOpenImage2 */
    
    double start = omp_get_wtime();
    
    cl_cmdqueue.enqueueNDRangeKernel(cl_soa_kernel_subtract_bg_and_threshold,
                                     cl::NullRange, cl::NDRange(width, height),
                                     cl::NDRange(local_x, local_y), NULL, &cl_event);
    cl_event.wait();
    
    double end = omp_get_wtime();
    cout << "time for soa subAndThresh kernel: " << end-start << endl;
    
    if(debug >= 2) {
        printf("--- INFO --- subtractBackgroundAndThreshold() - Frame %d: Subtracted background and performed thresholding.\n",frameCount);
    }
    
    /* Test output, read region values */
    // cl_cmdqueue.enqueueReadBuffer(cl_region_values,CL_TRUE, 0, sizeof(int)*size, int_image, NULL, &cl_event);
    // cl_event.wait();
    
    double end1 = omp_get_wtime();
    return end1-start1;
}

double SilhouetteExtraction::subtractBgThresh(int local_x, int local_y)
{
    double start1 = omp_get_wtime();
    
    cl::Event cl_event;
    
//    cl_uint_image = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(uint8_t)));
    cl_kernel_subtract_bg_and_threshold.setArg(3, image2d_fgmask);
    cl_kernel_subtract_bg_and_threshold.setArg(4, width);
    cl_kernel_subtract_bg_and_threshold.setArg(5, cl_region_values);
    
    /* For closeAndOpenImage2 -  TODO: remove when done */
    cl_kernel_subtract_bg_and_threshold.setArg(6, cl_fgmask);
    /* End closeAndOpenImage2 */
    
    double start = omp_get_wtime();
    
    cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_subtract_bg_and_threshold,
                                     cl::NullRange, cl::NDRange(width, height),
                                     cl::NDRange(local_x, local_y), NULL, &cl_event);
    cl_event.wait();
    
    double end = omp_get_wtime();
    cout << "time for def subAndThresh kernel: " << end-start << endl;
    
    if(debug >= 2) {
        printf("--- INFO --- subtractBackgroundAndThreshold() - Frame %d: Subtracted background and performed thresholding.\n",frameCount);
    }
    
    /* Test output, read region values */
    // cl_cmdqueue.enqueueReadBuffer(cl_region_values,CL_TRUE, 0, sizeof(int)*size, int_image, NULL, &cl_event);
    // cl_event.wait();

    double end1 = omp_get_wtime();
    return end1-start1;
}

/* Perform background subtraction and thresholding of the image,
 for initial foreground detection.
 TODO: we will need more kernels for the morphological processes,
 connected components labeling and the actual "silhouette extraction"
 with the "elastic spring" approach. TODO */
void SilhouetteExtraction::subtractBackgroundAndThreshold(int workgroupSize)
{
    cout << "--- SILHOUETTE EXTRACTION --- Subtract Background and Threshold" << endl;
    
    double time1, time2;
    
    int local_x, local_y;
    
    if (type == 0) {
        local_x = workgroupSize;
        local_y = 1;
    }
    else if (type == 1){
        local_x = 1;
        local_y = workgroupSize;
    }
    else{
        local_x = workgroupSize;
        local_y = workgroupSize;
    }

    try {
        time1 = soaSubtractBgThresh(local_x, local_y);
        time2 = subtractBgThresh(local_x, local_y);
    } catch (cl::Error cl_err) {
        printf("--- ERROR --- subtractBackgroundAndThreshold --- %s (%s)\n", cl_err.what(), oclErrorString(cl_err.err()));

        if (cl_err.err() == CL_BUILD_PROGRAM_FAILURE) {
            cl::STRING_CLASS buildLog;
            cl_program.getBuildInfo(cl_devices[1], CL_PROGRAM_BUILD_LOG, &buildLog);
            cout << buildLog << endl;
        }
        exit(1);
    }
}

/* Call the dilate/erode kernels for closing/opening the image */
void SilhouetteExtraction::closeAndOpenImage(int workgroupSize) {
    cl::Event cl_event;
    
    int local_x, local_y;
    
    if (type == 0) {
        local_x = workgroupSize;
        local_y = 1;
    }
    else if (type == 1){
        local_x = 1;
        local_y = workgroupSize;
    }
    else{
        local_x = workgroupSize;
        local_y = workgroupSize;
    }
    
    try {
        /* Start closing/opening */
        // size_t localsize = (local_xy+2)*(local_xy+2)*sizeof(uint32_t);

        // 1. Dilate
        cl_kernel_dilate.setArg(0, image2d_fgmask);
        cl_kernel_dilate.setArg(1, image2d_fgmask2);
        // cl_kernel_dilate.setArg(2,cl::__local(localsize));

        cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_dilate,
                                         cl::NullRange, cl::NDRange(width, height),
                                         cl::NDRange(local_x, local_y), NULL, &cl_event);
        cl_event.wait();

        /* Result now in image2d_fgmask2 */

        /* 2. Erode */
        cl_kernel_erode.setArg(0, image2d_fgmask2);
        cl_kernel_erode.setArg(1, image2d_fgmask);
        // cl_kernel_erode.setArg(2,cl::__local(localsize));

        cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_erode,
                                         cl::NullRange, cl::NDRange(width, height),
                                         cl::NDRange(local_x, local_y), NULL, &cl_event);
        cl_event.wait();

        /* Result now in image2d_fgmask */

        /* 3. Erode again */
        cl_kernel_erode.setArg(0, image2d_fgmask);
        cl_kernel_erode.setArg(1, image2d_fgmask2);
        // cl_kernel_erode.setArg(2,cl::__local(localsize));

        cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_erode,
                                         cl::NullRange, cl::NDRange(width, height),
                                         cl::NDRange(local_x, local_y), NULL, &cl_event);
        cl_event.wait();

        /* Result now in image2d_fgmask2 */

        // /* 4. Dilate again */
        cl_kernel_dilate.setArg(0, image2d_fgmask2);
        cl_kernel_dilate.setArg(1, image2d_fgmask);
        // cl_kernel_dilate.setArg(2,cl::__local(localsize));

        cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_dilate,
                                         cl::NullRange, cl::NDRange(width, height),
                                         cl::NDRange(local_x, local_y), NULL, &cl_event);
        cl_event.wait();

        /* Result now in image2d_fgmask */
        /* End Closing/Opening */
        
    } catch (cl::Error cl_err) {
        printf("--- ERROR --- closeAndOpenImage --- %s (%s)\n",cl_err.what(),oclErrorString(cl_err.err()));

        if(cl_err.err() == CL_BUILD_PROGRAM_FAILURE) {
            cl::STRING_CLASS buildLog;
            cl_program.getBuildInfo(cl_devices[1], CL_PROGRAM_BUILD_LOG, &buildLog);
            cout << buildLog << endl;
        }
        exit(1);
    }

    if(debug >=2) {
        printf("--- INFO --- closeAndOpenImage() - Frame %d: Performed closing and opening.\n",
        frameCount);
    }
}

/* START closeAndOpenImage2 - global mem */
/* Call the dilate/erode kernels for closing/opening the image */
void SilhouetteExtraction::closeAndOpenImage2(int workgroupSize) {
    cl::Event cl_event;
    
    int local_x;
    int local_y;
    
    if (type == 0) {
        local_x = workgroupSize;
        local_y = 1;
    }
    else if (type == 1){
        local_x = 1;
        local_y = workgroupSize;
    }
    else{
        local_x = workgroupSize;
        local_y = workgroupSize;
    }
    
    try {
        /* Start closing/opening */
//        int local_xy = 16;

        // 1. Dilate
        cl_kernel_dilate_2.setArg(0,cl_fgmask);
        cl_kernel_dilate_2.setArg(1,cl_fgmask2);
        // cl_kernel_dilate.setArg(2,cl::__local(localsize));

        cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_dilate_2,
        cl::NullRange, cl::NDRange(width,height),
        cl::NDRange(local_x, local_y), NULL, &cl_event);
        cl_event.wait();

        /* Result now in image2d_fgmask2 */

        /* 2. Erode */
        cl_kernel_erode_2.setArg(0,cl_fgmask2);
        cl_kernel_erode_2.setArg(1,cl_fgmask);
        // cl_kernel_erode.setArg(2,cl::__local(localsize));

        cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_erode_2,
        cl::NullRange, cl::NDRange(width,height),
        cl::NDRange(local_x, local_y), NULL, &cl_event);
        cl_event.wait();

        /* Result now in image2d_fgmask */

        /* 3. Erode again */
        cl_kernel_erode_2.setArg(0,cl_fgmask);
        cl_kernel_erode_2.setArg(1,cl_fgmask2);
        // cl_kernel_erode.setArg(2,cl::__local(localsize));

        cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_erode_2,
        cl::NullRange, cl::NDRange(width,height),
        cl::NDRange(local_x, local_y), NULL, &cl_event);
        cl_event.wait();

        /* Result now in image2d_fgmask2 */

        // /* 4. Dilate again */
        cl_kernel_dilate_2.setArg(0,cl_fgmask2);
        cl_kernel_dilate_2.setArg(1,cl_fgmask);
        // cl_kernel_dilate.setArg(2,cl::__local(localsize));

        cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_dilate_2,
        cl::NullRange, cl::NDRange(width,height),
        cl::NDRange(local_x, local_y), NULL, &cl_event);
        cl_event.wait();

        /* Result now in cl_fgmask */

        /* Test the output to see if it looks correct */
        // cl_cmdqueue.enqueueReadBuffer(cl_fgmask,CL_TRUE, 0, 
        // 				  sizeof(uint8_t)*size, outTest2, NULL, &cl_event);
        // cl_event.wait();

        // for(int i=0;i<size;i++) {
        //   if(outTest2[i] > 0) {
        // 	outTest2[i] = 255;
        //   }
        //   else {
        // 	outTest2[i] = 0;
        //   }
        // }
        /* End testing */
    } catch (cl::Error cl_err) {
        printf("--- ERROR --- closeAndOpenImage2 --- %s (%s)\n",cl_err.what(),oclErrorString(cl_err.err()));

        if(cl_err.err() == CL_BUILD_PROGRAM_FAILURE) {
            cl::STRING_CLASS buildLog;
            cl_program.getBuildInfo(cl_devices[1], CL_PROGRAM_BUILD_LOG, &buildLog);
            cout << buildLog << endl;
        }
        exit(1);
    }

    if(debug >=2) {
        printf("--- INFO --- closeAndOpenImage2() - Frame %d: Performed closing and opening.\n",
        frameCount);
    }
}
/* END CLOSEANDOPENIMAGE2 - global mem */

/* START closeAndOpenImage3 - all in one kernel */
void SilhouetteExtraction::closeAndOpenImage3(int workgroupSize) {
    cl::Event cl_event;
    
//    int local_x;
//    int local_y;
//    
//    if (type == 0) {
//        local_x = workgroupSize;
//        local_y = 1;
//    }
//    else if (type == 1){
//        local_x = 1;
//        local_y = workgroupSize;
//    }
//    else{
//        local_x = workgroupSize;
//        local_y = workgroupSize;
//    }
    
    try {
        /* Start closing/opening */

        /* Work group size - change when testing */
        int local_xy = 16;

        size_t localsize = (local_xy+2)*(local_xy+2)*sizeof(uint8_t);

        // Dilate -> Erode -> Erode -> Dilate, in one kernel
        cl_kernel_closeandopen.setArg(0,cl_fgmask);
        cl_kernel_closeandopen.setArg(1,cl_fgmask2);
        cl_kernel_closeandopen.setArg(2,cl::__local(localsize));

        cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_closeandopen,
        cl::NullRange, cl::NDRange(width,height),
        cl::NDRange(local_xy, local_xy), NULL, &cl_event);
        cl_event.wait();

        /* Result now in cl_fgmask2 */

        /* Test the output to see if it looks correct. */
        // cl_cmdqueue.enqueueReadBuffer(cl_fgmask2,CL_TRUE, 0, 
        // 				  sizeof(uint8_t)*size, outTest2, NULL, &cl_event);
        // cl_event.wait();

        // for(int i=0;i<size;i++) {
        //   if(outTest2[i] > 0) {
        // 	outTest2[i] = 255;
        //   }
        //   else {
        // 	outTest2[i] = 0;
        //   }
        // }
        /* End testing */
    } catch (cl::Error cl_err) {
        printf("--- ERROR --- closeAndOpenImage3 --- %s (%s)\n",cl_err.what(),oclErrorString(cl_err.err()));

        if(cl_err.err() == CL_BUILD_PROGRAM_FAILURE) {
            cl::STRING_CLASS buildLog;
            cl_program.getBuildInfo(cl_devices[1], CL_PROGRAM_BUILD_LOG, &buildLog);
            cout << buildLog << endl;
        }
        exit(1);
    }

    if(debug >=2) {
        printf("--- INFO --- closeAndOpenImage3() - Frame %d: Performed closing and opening.\n",
        frameCount);
    }
}
/* END closeAndOpenImage3 - all in one kernel */

/* Perform Connected Components Labeling (CCL) on binary image */
void SilhouetteExtraction::labelConnectedComponents(int workgroupSize) {
    cl::Event cl_event;

    int scanning_iterations = 0;
    image2d_fgmask2 = cl::Image2D(cl_context, CL_MEM_READ_WRITE, uint32_format, width, height, 0, NULL);
    int local_x;
    int local_y;
    
    if (type == 0) {
        local_x = workgroupSize;
        local_y = 1;
    }
    else if (type == 1){
        local_x = 1;
        local_y = workgroupSize;
    }
    else{
        local_x = workgroupSize;
        local_y = workgroupSize;
    }

    try {
        cl_kernel_ccl_init_labels.setArg(0,image2d_fgmask);
        cl_kernel_ccl_init_labels.setArg(1,image2d_fgmask2);

        cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_ccl_init_labels,
                                         cl::NullRange, cl::NDRange(width, height),
                                         cl::NDRange(local_x, local_y), NULL, &cl_event);
        cl_event.wait();

        image2d_fgmask = cl::Image2D(cl_context, CL_MEM_READ_WRITE, uint32_format, width, height, 0, NULL);
        /* We need uint32_t for this */
        cl_uint_image = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(uint32_t)));

        /* Buffer for isNotDone saving: */
        cl_uint_val = cl::Buffer(cl_context, CL_MEM_READ_WRITE, sizeof(uint32_t));

        /* Set 0 in cl_uint_image before starting */
        // TODO: this should be done in the kernel instead. (e.g. if x == 0 && y== 0{isNotDone[0] = 0})
        uint32_image[0] = 0;
        cl_cmdqueue.enqueueWriteBuffer(cl_uint_image, CL_TRUE, 0, sizeof(uint8_t), outTest2, NULL, NULL);

        size_t localsize = 18*18*sizeof(uint32_t);

        while(1) {
            scanning_iterations++;

            /* Make sure we always assign the output buffer as input
            and input as output for each iteration */
            if((scanning_iterations % 2) == 1) {
                cl_kernel_ccl_scanning.setArg(0, image2d_fgmask2);
                cl_kernel_ccl_scanning.setArg(1, image2d_fgmask);
                //	cl_kernel_ccl_scanning.setArg(4,cl::__local(localsize));
            }
            else {
                cl_kernel_ccl_scanning.setArg(0, image2d_fgmask);
                cl_kernel_ccl_scanning.setArg(1, image2d_fgmask2);
                //	cl_kernel_ccl_scanning.setArg(4,cl::__local(localsize));
            }
            cl_kernel_ccl_scanning.setArg(2, cl_uint_val);
            cl_kernel_ccl_scanning.setArg(3, cl_uint_image);

            cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_ccl_scanning,
                                             cl::NullRange, cl::NDRange(width, height),
                                             cl::NDRange(local_x, local_y), NULL, &cl_event);
            cl_event.wait();

            cl_cmdqueue.enqueueReadBuffer(cl_uint_val, CL_TRUE, 0, sizeof(uint32_t), uint32_image, NULL, &cl_event);
            cl_event.wait();

            if(uint32_image[0] == 0) {
                break;
            }
            else {
                /* Set 0 in cl_uint_image */
                uint32_image[0] = 0;
                //	cl_cmdqueue.enqueueWriteBuffer(cl_uint_image,CL_TRUE,0,sizeof(uint8_t),outTest2, NULL, NULL);
            }
        }

        if(debug >=2) {
            printf("--- INFO --- labelConnectedComponents() - Frame %d: Scanning iterations: %d.\n",
            frameCount, scanning_iterations);
        }

        /* Last step - analysis */
        /* Test */

        //    Set origin and region. Read whole image
        // cl::size_t<3> origin;
        // origin[0] = 0; origin[1] = 0, origin[2] = 0;
        // cl::size_t<3> region;
        // region[0] = width; region[1] = height; region[2] = 1;

        // if((scanning_iterations %2) == 1) {
        //   cl_cmdqueue.enqueueReadImage(image2d_fgmask, CL_TRUE, origin, region,
        // 				   0,0,uint32_image,NULL,&cl_event);
        // }
        // else {
        //   cl_cmdqueue.enqueueReadImage(image2d_fgmask2, CL_TRUE, origin, region,
        // 				   0,0,uint32_image,NULL,&cl_event);

        // }
        // cl_event.wait();

        // for(int i=0;i<size;i++) {
        //   if(uint32_image[i] > 0) {
        // 	printf("blah: %d\n",uint32_image[i]);
        //   }
        // }

        /* End test */


        if((scanning_iterations % 2) == 1)
            cl_kernel_ccl_analysis.setArg(0,image2d_fgmask);
        else
            cl_kernel_ccl_analysis.setArg(0,image2d_fgmask2);

        cl_kernel_ccl_analysis.setArg(1,cl_uint_image);
        cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_ccl_analysis,
                                         cl::NullRange, cl::NDRange(width, height),
                                         cl::NDRange(local_x, local_y), NULL, &cl_event);
        cl_event.wait();

        /* USED FOR DISPLAYING TEST-DATA */
        // comment out when performing tests
//         cl_cmdqueue.enqueueReadBuffer(cl_uint_image,CL_TRUE, 0, sizeof(uint32_t)*size, uint32_image, NULL, &cl_event);
//         cl_event.wait();

        //for(int i=0;i<size;i++) {
        //   if(uint32_image[i] > 0) {
	//	cout << uint32_image[i] << endl;
        // 	outTest2[i] = 255;
           //}
           //else {
         //	outTest2[i] = 0;
	//	cout << uint32_image[i] << endl;
           //}
           //outTest2[i] = (uint8_t) uint32_image[i];
         //}
	
//	cv::Mat outImage = cv::Mat(height, width, CV_8UC1);
//	outImage.data = (uchar*) uint32_image;
//	cv::imwrite("silhouette.jpg", outImage);
        /* END TEST-DISPLAY */

        if(debug >=2) {
            printf("--- INFO --- labelConnectedComponents() - Frame %d: Analysis step done.\n",
            frameCount);
        }

    }
    catch (cl::Error cl_err) {
        printf("--- ERROR --- labelConnectedComponents --- %s (%s)\n",cl_err.what(),oclErrorString(cl_err.err()));

        if(cl_err.err() == CL_BUILD_PROGRAM_FAILURE) {
            cl::STRING_CLASS buildLog;
            cl_program.getBuildInfo(cl_devices[1], CL_PROGRAM_BUILD_LOG, &buildLog);
            cout << buildLog << endl;
        }
        exit(1);
    }

    if(debug >=2) {
        printf("--- INFO --- labelConnectedComponents() - Frame %d: Labeled connected components\n",
        frameCount);
    }  
}

/* Perform Connected Components Labeling (CCL) on binary image.
* Algorithm from GPU Computing Gems book.
* Input (for first kernel):
* For now, cl_uint_image2 (uint8_t) from closeAndOpenImage kernel.
* This should be removed, as we do a read from Image2D to Buffer in that
* function. TODO.
* NOTE: this doesn't work correctly now. Execution of kernels etc.
* are based on reading the example code from here: http://www.gpucomputing.net/?q=node/1312
* To get it working, a more thorough reading and implementation of this code
* could be done.
*/
void SilhouetteExtraction::labelConnectedComponents2(int workgroupSize) {
    cl::Event cl_event;

    int local_x = 16;
    int local_y = 16;
    //	cl_kernel_ccl_scanning.setArg(4,cl::__local(localsize));
    
//    int local_x;
//    int local_y;
    
//    if (type == 0) {
//        local_x = workgroupSize;
//        local_y = 1;
//    }
//    else if (type == 1){
//        local_x = 1;
//        local_y = workgroupSize;
//    }
//    else{
//        local_x = workgroupSize;
//        local_y = workgroupSize;
//    }

    size_t localSize = (local_x+2)*(local_y+2)*sizeof(uint8_t);
    size_t localSize_int = (local_x+2)*(local_y+2)*sizeof(int);

    /* Kernel 1 */
    try {
        // 32-bit uint
        //cl_uint_image = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(uint32_t)));
        // changed, now cl_int_image with regular int
        cl_int_image = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(int)));

        cl_kernel_ccl2_1.setArg(0,cl_uint_image2); // input: output from close/open - uint
        cl_kernel_ccl2_1.setArg(1,cl_int_image); // output: int
        cl_kernel_ccl2_1.setArg(2,cl::__local(localSize)); // local uint BLOCK_WIDTH_PAD^2
        cl_kernel_ccl2_1.setArg(3,cl::__local(localSize_int)); // local uint BLOCK_WIDTH_PAD^2
        cl_kernel_ccl2_1.setArg(4,cl::__local((size_t)sizeof(uint32_t))); // local uint size 1
        cl_kernel_ccl2_1.setArg(5,width); // img width

        cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_ccl2_1,
        cl::NullRange, cl::NDRange(width, height),
        cl::NDRange(local_x, local_y), NULL, &cl_event);
        cl_event.wait();

        if(debug >=3) {
            printf("--- DEBUG --- labelConnectedComponents2() - Frame %d: Kernel 1 done.\n",
            frameCount);
        }

    }
    catch (cl::Error cl_err) {
        printf("--- ERROR --- labelConnectedComponents2 --- Kernel 1 --- %s (%s)\n",cl_err.what(),oclErrorString(cl_err.err()));

        if(cl_err.err() == CL_BUILD_PROGRAM_FAILURE) {
            cl::STRING_CLASS buildLog;
            cl_program.getBuildInfo(cl_devices[1], CL_PROGRAM_BUILD_LOG, &buildLog);
            cout << buildLog << endl;
        }
        exit(1);
    }

    /* End kernel 1 */

    /* Testing output from kernel 1 */
    // cl_cmdqueue.enqueueReadBuffer(cl_int_image,CL_TRUE, 0, 
    // 				size*sizeof(int), int_image, NULL, &cl_event);
    // cl_event.wait();

    // for(int i=0;i<size;i++) {
    //   if(int_image[i] != -1) {
    //     outTest2[i] = 255;
    //   }
    //   else {
    //     outTest2[i] = 0;
    //   }
    // }

    /* End testing output from kernel 1 */

    int log2width = log2(width);
    int threadsX = local_x; // TODO!
    int threadsY = local_y; // same same as above. What to choose here?
    int tileSize = threadsX;//*threadsY;

    // /* Kernel 2, 3 and 4 */
    try {
        cl_kernel_ccl2_2.setArg(0,cl_int_image); // input is output from kernel 1
        // image2d_fgmask is output from closeAndOpen. same data as in cl_uint_image2
        // (the buffer kernel 1 gets). TODO: let kernel1 use image2d_fgmask too, instead.
        // then test if the runtime differs.... !
        
        cl_kernel_ccl2_2.setArg(1,image2d_fgmask);
        //cl_kernel_ccl2_2.setArg(1,cl_uint_image2); // test
        cl_kernel_ccl2_2.setArg(2,cl::__local((size_t)sizeof(uint32_t)));
        cl_kernel_ccl2_2.setArg(3,width);

        // /* Kernel 3 args */
        // // same buffer for in and out? Seems so -
        // // why not use only one arg for it? example code has it like this. TODO
        cl_kernel_ccl2_3.setArg(0, cl_int_image);
        cl_kernel_ccl2_3.setArg(1, image2d_fgmask);
        //cl_kernel_ccl2_3.setArg(1, cl_uint_image2);
        cl_kernel_ccl2_3.setArg(2, cl::__local((size_t)sizeof(uint32_t)));
        cl_kernel_ccl2_3.setArg(3, width);
        cl_kernel_ccl2_3.setArg(4, log2width);
        // /* End kernel 3 args, (more below) */      

        while(tileSize < width || tileSize < height) {
            /* Compute number of tiles that are going to be merged
            in a single thread block */
            int xTiles = 4;
            int yTiles = 4;
            
            if(xTiles*tileSize > width) {
                xTiles = width / tileSize;
            }
            
            if(yTiles*tileSize > height) {
                yTiles = height / tileSize;
            }
            
            //      printf("xtiles: %d, ytiles: %d\n",xTiles,yTiles);

            /* Number of threads used to merge neighboring tiles */
            int threadsPerBlock = 32;
            if(tileSize < threadsPerBlock) {
                threadsPerBlock = tileSize;
            }

            // printf("threadsPerBlock: %d\n",threadsPerBlock);

            // printf("width: %d\n",width);
            // printf("height: %d\n",height);
            // printf("tilesize: %d\n",tileSize);

            cl::NDRange local_range(xTiles, yTiles, threadsPerBlock);
            cl::NDRange global_range(width, height);
            //cl::NDRange global_range(width, height, 1);

            cl_kernel_ccl2_2.setArg(4, tileSize); //?why this as "tileSize"? took from example code - TODO

            cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_ccl2_2, cl::NullRange, global_range, local_range, NULL, &cl_event);
            cl_event.wait();

            if(debug >= 3) {
                printf("--- DEBUG --- labelConnectedComponents2() - Frame %d: Kernel 2 done.\n",
                frameCount);
            }

            /* TEST ! */
            // cl_cmdqueue.enqueueReadBuffer(cl_uint_val,CL_TRUE, 0,
            // 				    sizeof(uint32_t), uint32_image, NULL, &cl_event);
            // cl_event.wait();
            //   /* END TEST! */

            if(yTiles > xTiles)
                tileSize = yTiles * tileSize;
            else
                tileSize = xTiles * tileSize;

            if(tileSize < width || tileSize < height) {
                // Update borders (kernel 3)
                /* Prepare */
                int threadsXtmp = threadsX;
                int threadsYtmp = threadsY;
                int tileX = width / tileSize;
                int tileY = height / tileSize;
                int maxThreads = threadsXtmp * threadsYtmp;
                
                if(tileY < threadsYtmp) {
                    threadsYtmp = tileY;
                    threadsXtmp = maxThreads / threadsYtmp;
                }
                
                if(threadsXtmp > tileSize)
                    threadsXtmp = tileSize;
                
                cl::NDRange block(threadsXtmp, threadsYtmp);
                //      cl::NDRange grid(width, height);
                cl::NDRange grid(width, tileY);
                int blocksPerTile = tileSize / threadsXtmp;
                
                /* End prepare */
                cl_kernel_ccl2_3.setArg(5, tileSize);
                cl_kernel_ccl2_3.setArg(6, blocksPerTile);

                cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_ccl2_3, cl::NullRange, grid, block, NULL,&cl_event);
                cl_event.wait();
            }

            if(debug >= 3) {
                printf("--- DEBUG --- labelConnectedComponents2() - Frame %d: Kernel 3 done.\n",
                frameCount);
            }
        }

        /* Kernel 4 */
        // cl_kernel_ccl2_4.setArg(0,cl_int_image);
        // cl_kernel_ccl2_4.setArg(1,width);
        // cl_kernel_ccl2_4.setArg(2,log2width);

        // cl::NDRange grid(width,height);
        // cl::NDRange block(local_x, local_y);

        // cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_ccl2_4,
        // 				     cl::NullRange,
        // 				     grid,
        // 				     block,
        // 				     NULL, &cl_event);
        // printf("Kernel 4 done\n");

        /* End kernel 4 */

        /* TEST ! */
        cl_cmdqueue.enqueueReadBuffer(cl_int_image,CL_TRUE, 0, size*sizeof(int), int_image, NULL, &cl_event);
        cl_event.wait();

        for(int i=0;i<size;i++) {
            if(int_image[i] > 0)
                outTest2[i] = 255;
            else
                outTest2[i] = 0;
            
            // outTest2[i] = (uint8_t) int_image[i];
        }
        /* End TEST! */
    }
    catch (cl::Error cl_err) {
        printf("--- ERROR --- labelConnectedComponents2 --- %s (%s)\n",cl_err.what(),oclErrorString(cl_err.err()));

        if(cl_err.err() == CL_BUILD_PROGRAM_FAILURE) {
            cl::STRING_CLASS buildLog;
            cl_program.getBuildInfo(cl_devices[1], CL_PROGRAM_BUILD_LOG, &buildLog);
            cout << buildLog << endl;
        }
        exit(1);
    }
    /* End kernel 2 */

    if(debug >=2) {
        printf("--- INFO --- labelConnectedComponents() - Frame %d: Labeled connected components\n",
        frameCount);
    }  
}
/* End labelConnectedComponents2 */

double SilhouetteExtraction::soaUpdateBackground(int local_x, int local_y)
{
    double start1 = omp_get_wtime();
    
    cl::Event cl_event;
    
    cl_soa_kernel_update_bg.setArg(0, image2d_fgmask);
    cl_soa_kernel_update_bg.setArg(1, cl_Data1);
    cl_soa_kernel_update_bg.setArg(2, cl_Data2);
    cl_soa_kernel_update_bg.setArg(3, cl_Mean_and_Variance);
    cl_soa_kernel_update_bg.setArg(4, cl_Mean_and_Variance);
    cl_soa_kernel_update_bg.setArg(5, cl_Mean_and_Variance);
    cl_soa_kernel_update_bg.setArg(6, cl_Mean_and_Variance);
    
    double start = omp_get_wtime();
    
    cl_cmdqueue.enqueueNDRangeKernel(cl_soa_kernel_update_bg,
                                     cl::NullRange, cl::NDRange(width, height),
                                     cl::NDRange(local_x, local_y), NULL, &cl_event);
    cl_event.wait();
    
    double end = omp_get_wtime();
    cout << "time for soa updateBg kernel: " << end-start << endl;
    
    double end1 = omp_get_wtime();
    return end1-start1;
}

double SilhouetteExtraction::updateBackground(int local_x, int local_y)
{
    double start1 = omp_get_wtime();
    
    cl::Event cl_event;
    
    cl_kernel_update_bg.setArg(0, image2d_fgmask);
    cl_kernel_update_bg.setArg(1, cl_Data);
    cl_kernel_update_bg.setArg(2, cl_Mean_and_Variance);
    
    double start = omp_get_wtime();
    
    cl_cmdqueue.enqueueNDRangeKernel(cl_kernel_update_bg,
                                     cl::NullRange, cl::NDRange(width, height),
                                     cl::NDRange(local_x, local_y), NULL, &cl_event);
    cl_event.wait();
    
    double end = omp_get_wtime();
    cout << "time for def updateBg kernel: " << end-start << endl;
    
    double end1 = omp_get_wtime();
    return end1-start1;
}

/* Call kernel for updating BG after all processing has been done.
Updates mean and variance for BG regions.
Input is the segmented image (image2d) where background is
0, foreground is > 0, cl_mean_and_variance, cl_Data */
void SilhouetteExtraction::updateBG(int workgroupSize)
{
    cout << "--- SILHOUETTE EXTRACTION --- Update Background" << endl;
    
    double time1, time2;
    int local_x, local_y;
    
    if (type == 0) {
        local_x = workgroupSize;
        local_y = 1;
    }
    else if (type == 1){
        local_x = 1;
        local_y = workgroupSize;
    }
    else{
        local_x = workgroupSize;
        local_y = workgroupSize;
    }

    try {
        time1 = soaUpdateBackground(local_x, local_y);
        time2 = updateBackground(local_x, local_y);
    }
    catch (cl::Error cl_err) {
        printf("--- ERROR --- updateBG --- %s (%s)\n", cl_err.what(), oclErrorString(cl_err.err()));

        if (cl_err.err() == CL_BUILD_PROGRAM_FAILURE) {
            cl::STRING_CLASS buildLog;
            cl_program.getBuildInfo(cl_devices[1], CL_PROGRAM_BUILD_LOG, &buildLog);
            cout << buildLog << endl;
        }
        exit(1);
    }

    if(debug >=2) {
        printf("--- INFO --- updateBG() - Frame %d: Updated mean and variance for BG.\n",
        frameCount);
    }
}


/* Setup all the buffers that are shared and reused 
for each new frame during silhouette extraction. Run once. */
void SilhouetteExtraction::setupBuffersForExtraction()
{
    try {
        /* Create kernels */
        cl_soa_kernel_convert_rgb_img = cl::Kernel(cl_program, "soa_convert_rgb_to_h_and_y");
        cl_kernel_convert_rgb_img = cl::Kernel(cl_program, "convert_rgb_to_h_and_y");
        
        cl_soa_kernel_mean_variance_kurtosis = cl::Kernel(cl_program, "soa_calc_mean_variance_kurtosis");
        cl_kernel_mean_variance_kurtosis = cl::Kernel(cl_program, "calc_mean_variance_kurtosis");
        
        cl_soa_kernel_update_bg_model = cl::Kernel(cl_program, "soa_update_bg_model");
        cl_kernel_update_bg_model = cl::Kernel(cl_program, "update_bg_model");
        
        cl_soa_kernel_subtract_bg_and_threshold = cl::Kernel(cl_program, "soa_subtract_and_threshold");
        cl_kernel_subtract_bg_and_threshold = cl::Kernel(cl_program, "subtract_and_threshold");
        
        cl_kernel_dilate = cl::Kernel(cl_program, "dilate_image");
        cl_kernel_erode = cl::Kernel(cl_program, "erode_image");

        /* For global mem - remove later. TODO */ 
        cl_kernel_dilate_2 = cl::Kernel(cl_program, "dilate_image_globalmem");
        cl_kernel_erode_2 = cl::Kernel(cl_program, "erode_image_globalmem");

        /* For closeAndOpen in one kernel */
        cl_kernel_closeandopen = cl::Kernel(cl_program, "close_and_open_image");

        /* CCL algorithm 1 */
        cl_kernel_ccl_init_labels = cl::Kernel(cl_program, "CCL_init_labels");
        cl_kernel_ccl_analysis = cl::Kernel(cl_program, "CCL_analysis");
        cl_kernel_ccl_scanning = cl::Kernel(cl_program, "CCL_scanning");
        cl_kernel_ccl_calculate_size = cl::Kernel(cl_program, "CCL_calculate_size");

        /* CCL algorithm 2 */
        cl_kernel_ccl2_1 = cl::Kernel(cl_program, "CCL_kernel_1");
        cl_kernel_ccl2_2 = cl::Kernel(cl_program, "CCL_kernel_2");
        cl_kernel_ccl2_3 = cl::Kernel(cl_program, "CCL_kernel_3");
        cl_kernel_ccl2_4 = cl::Kernel(cl_program, "CCL_kernel_4");

        /* Update BG kernel */
        cl_soa_kernel_update_bg = cl::Kernel(cl_program, "soa_background_update");
        cl_kernel_update_bg = cl::Kernel(cl_program, "background_update");

        /* Data that is reused between frames */
        cl_Data = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*2*sizeof(int)));
        cl_Data1 = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(int)));
        cl_Data2 = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(int)));
        
        cl_Model = cl::Buffer(cl_context, CL_MEM_READ_WRITE,(size*2*sizeof(float)));
        cl_Model1 = cl::Buffer(cl_context, CL_MEM_READ_WRITE,(size*sizeof(float)));
        cl_Model2 = cl::Buffer(cl_context, CL_MEM_READ_WRITE,(size*sizeof(float)));
        
        cl_DistMap = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*2*sizeof(int)));
        cl_DistMap1 = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(int)));
        cl_DistMap2 = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(int)));
        
        cl_Mean_and_Variance = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*4*sizeof(float)));
        cl_Mean_and_Variance1 = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(float)));
        cl_Mean_and_Variance2 = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(float)));
        cl_Mean_and_Variance3 = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(float)));
        cl_Mean_and_Variance4 = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(float)));
        
        cl_M2 = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*2*sizeof(float)));
        cl_M21 = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(float)));
        cl_M22 = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(float)));
        
        cl_M3 = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*2*sizeof(float)));
        cl_M31 = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(float)));
        cl_M32 = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(float)));
        
        cl_M4 = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*2*sizeof(float)));
        cl_M41 = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(float)));
        cl_M42 = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(float)));
        
        bin_img_format.image_channel_order = CL_R;
        bin_img_format.image_channel_data_type = CL_UNSIGNED_INT8;

        image2d_fgmask = cl::Image2D(cl_context, CL_MEM_READ_WRITE, bin_img_format, width, height, 0, NULL);
        image2d_fgmask2 = cl::Image2D(cl_context, CL_MEM_READ_WRITE, bin_img_format, width, height, 0, NULL);

        /* For global mem close and open. TODO: remove later */
        cl_fgmask = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(uint8_t)));
        cl_fgmask2 = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(uint8_t)));
        /* END GLBLMEM closeAndOpen2 */

        uint32_format.image_channel_order = CL_R;
        uint32_format.image_channel_data_type = CL_UNSIGNED_INT32;
        cl_region_values = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (size*sizeof(int)));

        // image2d_ccl = cl::Image2D(cl_context, CL_MEM_READ_WRITE,
        // 			      uint32_format, width, height, 0, NULL);
        // image2d_ccl2 = cl::Image2D(cl_context, CL_MEM_READ_WRITE,
        // 			      uint32_format, width, height, 0, NULL);

        /* Set args that doesn't change */
        cl_soa_kernel_convert_rgb_img.setArg(3, cl_Data1);
        cl_soa_kernel_convert_rgb_img.setArg(4, cl_Data2);
        cl_soa_kernel_convert_rgb_img.setArg(5, width);
        
        cl_kernel_convert_rgb_img.setArg(1, cl_Data);
        cl_kernel_convert_rgb_img.setArg(2, width);

        cl_soa_kernel_mean_variance_kurtosis.setArg(3, cl_Data1);
        cl_soa_kernel_mean_variance_kurtosis.setArg(4, cl_Data2);
        cl_soa_kernel_mean_variance_kurtosis.setArg(5, cl_Mean_and_Variance1);
        cl_soa_kernel_mean_variance_kurtosis.setArg(6, cl_Mean_and_Variance2);
        cl_soa_kernel_mean_variance_kurtosis.setArg(7, cl_Mean_and_Variance3);
        cl_soa_kernel_mean_variance_kurtosis.setArg(8, cl_Mean_and_Variance4);
        cl_soa_kernel_mean_variance_kurtosis.setArg(9, cl_M21);
        cl_soa_kernel_mean_variance_kurtosis.setArg(10, cl_M22);
        cl_soa_kernel_mean_variance_kurtosis.setArg(11, cl_M31);
        cl_soa_kernel_mean_variance_kurtosis.setArg(12, cl_M32);
        cl_soa_kernel_mean_variance_kurtosis.setArg(13, cl_M41);
        cl_soa_kernel_mean_variance_kurtosis.setArg(14, cl_M42);
        cl_soa_kernel_mean_variance_kurtosis.setArg(15, cl_DistMap1);
        cl_soa_kernel_mean_variance_kurtosis.setArg(16, cl_DistMap2);
        cl_soa_kernel_mean_variance_kurtosis.setArg(17, width);
        
        cl_kernel_mean_variance_kurtosis.setArg(3, cl_Data);
        cl_kernel_mean_variance_kurtosis.setArg(4, cl_Mean_and_Variance);
        cl_kernel_mean_variance_kurtosis.setArg(5, cl_M2);
        cl_kernel_mean_variance_kurtosis.setArg(6, cl_M3);
        cl_kernel_mean_variance_kurtosis.setArg(7, cl_M4);
        cl_kernel_mean_variance_kurtosis.setArg(8, cl_DistMap);
        cl_kernel_mean_variance_kurtosis.setArg(9, width);

        cl_soa_kernel_update_bg_model.setArg(0, cl_Data1);
        cl_soa_kernel_update_bg_model.setArg(1, cl_Data2);
        cl_soa_kernel_update_bg_model.setArg(2, cl_Model1);
        cl_soa_kernel_update_bg_model.setArg(3, cl_Model2);
        cl_soa_kernel_update_bg_model.setArg(4, cl_Mean_and_Variance1);
        cl_soa_kernel_update_bg_model.setArg(5, cl_Mean_and_Variance2);
        cl_soa_kernel_update_bg_model.setArg(6, cl_Mean_and_Variance3);
        cl_soa_kernel_update_bg_model.setArg(7, cl_Mean_and_Variance4);
        cl_soa_kernel_update_bg_model.setArg(8, cl_DistMap1);
        cl_soa_kernel_update_bg_model.setArg(9, cl_DistMap2);
        
        cl_kernel_update_bg_model.setArg(0, cl_Data);
        cl_kernel_update_bg_model.setArg(1, cl_Model);
        cl_kernel_update_bg_model.setArg(2, cl_Mean_and_Variance);
        cl_kernel_update_bg_model.setArg(3, cl_DistMap);

        cl_soa_kernel_subtract_bg_and_threshold.setArg(0, cl_Data1);
        cl_soa_kernel_subtract_bg_and_threshold.setArg(1, cl_Data2);
        cl_soa_kernel_subtract_bg_and_threshold.setArg(2, cl_Model1);
        cl_soa_kernel_subtract_bg_and_threshold.setArg(3, cl_Model2);
        cl_soa_kernel_subtract_bg_and_threshold.setArg(4, cl_Mean_and_Variance1);
        cl_soa_kernel_subtract_bg_and_threshold.setArg(5, cl_Mean_and_Variance2);
        cl_soa_kernel_subtract_bg_and_threshold.setArg(6, cl_Mean_and_Variance3);
        cl_soa_kernel_subtract_bg_and_threshold.setArg(7, cl_Mean_and_Variance4);
        
        cl_kernel_subtract_bg_and_threshold.setArg(0, cl_Data);
        cl_kernel_subtract_bg_and_threshold.setArg(1, cl_Model);
        cl_kernel_subtract_bg_and_threshold.setArg(2, cl_Mean_and_Variance);
        
    }
    catch (cl::Error cl_err) {
        printf("--- ERROR --- setupBuffersForExtraction --- %s (%s)\n", cl_err.what(), oclErrorString(cl_err.err()));

        if(cl_err.err() == CL_BUILD_PROGRAM_FAILURE) {
            cl::STRING_CLASS buildLog;
            cl_program.getBuildInfo(cl_devices[1], CL_PROGRAM_BUILD_LOG, &buildLog);
            cout << buildLog << endl;
        }
        exit(1);
    }
}

void SilhouetteExtraction::setDebug(int debug_lvl) {
    debug = debug_lvl;
}

int SilhouetteExtraction::getDebug() {
    return debug;
}

int SilhouetteExtraction::getCurFrameNum() {
    return curFrameNum;
}

int SilhouetteExtraction::getFrameCount() {
    return frameCount;
}

/* Show the output on screen. For now using OpenCV
cvShowImage. Slow and painful. Better to copy frame
to graphics card and show directly, I suppose. TODO */
void SilhouetteExtraction::showOutput() {
    cv::Mat testImage = cv::Mat(height, width, CV_8UC1);
    cl::Event cl_event;
    cl_cmdqueue.enqueueReadBuffer(cl_uint_image,CL_TRUE, 0, sizeof(uint32_t)*size, uint32_image, NULL, &cl_event);
    cl_event.wait();

  	for(int i=0;i<size;i++) {
           if(uint32_image[i] > 0) {
                outTest2[i] = 255;
           } else {
                outTest2[i] = 0;
           }
        }

    testImage.data = (uchar*) outTest2;
    imwrite("silhouette.jpg", testImage);
}

void SilhouetteExtraction::printStats() {
}

void SilhouetteExtraction::avg(struct timeval start, struct timeval end, vector<float>& avg, int nin){
    float sum=0;
    float delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec);

    avg.push_back(delta);
    for (int ii = 0; ii < avg.size(); ii++){
        sum += avg[ii];
    }
    
    cout << "avg for instruction " << nin << ": " << (int)sum/avg.size() << endl;
}
