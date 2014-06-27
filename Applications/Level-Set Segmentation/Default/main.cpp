//#define __NO_STD_VECTOR
#define __CL_ENABLE_EXCEPTIONS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <GL/glew.h>
#include <GLUT/glut.h>
#include <omp.h>

#include <iostream>

#include <OpenCL/cl.hpp>

#include "../CL/oclUtils.h"

#define IMAGE		"input.bmp"
#define ITERATIONS   5000
#define THRESHOLD	 180
#define EPSILON		 45

#define BLOCKDIM_X	 32
#define BLOCKDIM_Y	 8

using namespace std;

typedef struct {
    unsigned char x, y, z, w;
} uchar4;

float *phi, *D;
uchar4 *h_Src, *h_Mask;
int imageW, imageH, N, pitch;
size_t pitchbytes;

float *d_phi, *d_D;
float *d_phi1;

void LoadBMPFile(uchar4 **dst, int *width, int *height, const char *name);
void sedt2d(int *_d,unsigned char *_bimg,int _h,int _w);

int its = 0;

int r;
int c;
int i;

void init_phi()
{

	int *init;
	unsigned char *mask;
	const char *mask_path = "input.bmp";
    
	if ((init = (int *)malloc(imageW*imageH*sizeof(int))) == NULL)
		printf("ME_INIT\n");

	if ((phi = (float *)malloc(imageW*imageH*sizeof(float))) == NULL)
		printf("ME_PHI\n");

	mask = (unsigned char *)malloc(imageW*imageH*sizeof(unsigned char));

	//printf("Init Mask\n");
	LoadBMPFile(&h_Mask, &imageW, &imageH, mask_path);

	for(r=0; r<imageH; r++)
    {
		for(c=0; c<imageW; c++)
        {
			mask[r*imageW + c] = (h_Mask[r*imageW + c].x)/255;
			//printf("%3d ", mask[r*imageW+c]);
		}
		//printf("\n");
	}

	sedt2d(init,mask,imageH,imageW);

	//printf("sdf of init mask\n");
	for (r=0; r<imageH; r++)
    {
		for (c=0; c<imageW; c++)
        {
			phi[r*imageW + c] = (float)init[r*imageW + c];
            
			if (phi[r*imageW + c] > 0)
				phi[r*imageW + c] = 0.5*sqrt(fabs(phi[r*imageW + c]));
            else
				phi[r*imageW + c] = -0.5*sqrt(fabs(phi[r*imageW + c]));
			//printf("%6.3f ", phi[r*imageW+c]);
		}
		//printf("\n");
	}

	free(init);
	free(mask);
}

int main(int argc, char** argv)
{
    int device = atoi(argv[1]);
    
	// Load the Input Image using BMPLoader
	const char *image_path = IMAGE;
	LoadBMPFile(&h_Src, &imageW, &imageH, image_path);
	D = (float *)malloc(imageW*imageH*sizeof(float));
    
	for (r=0; r<imageH; r++)
		for (c=0; c<imageW; c++)
			D[r*imageW + c] = h_Src[r*imageW + c].x;

	N = imageW*imageH;

	// Threshold based on hash defined paramters
	for(i=0; i<N; i++)
		D[i] = EPSILON - fabs(D[i] - THRESHOLD);

	// Init phi to SDF
	init_phi();
    
    size_t kernels_size;
    vector<cl::Platform> cl_platforms;
    vector<cl::Device> cl_devices;
    cl::Program cl_program;
    cl::Context cl_context;
    cl::Program::Sources cl_sources;
    cl::CommandQueue cl_cmdqueue;
    cl::Event cl_event;
    cl::Kernel cl_kernel;
    
    try {
        cl::Platform::get(&cl_platforms);

        if (cl_platforms.size() == 0) {
            printf("--- ERROR --- OpenCL platform-size is 0. Exiting.\n");
            exit(1);
        }

        cl_context_properties cl_properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(cl_platforms[0])(), 0};

        if (device == 0)
            cl_context = cl::Context(CL_DEVICE_TYPE_CPU, cl_properties);
        else
            cl_context = cl::Context(CL_DEVICE_TYPE_GPU, cl_properties);

        cl_devices = cl_context.getInfo<CL_CONTEXT_DEVICES>();
        //cout << "Using " << cl_devices.size() << " devices " << endl;

        vector<cl::Platform> all_platforms;
        cl::Platform::get(&all_platforms);
        
        if (all_platforms.size() == 0) {
            cout << " No platforms found. Check OpenCL installation!" << endl;
            exit(1);
        }
        
        cl::Platform default_platform = all_platforms[0];
        string platformName = default_platform.getInfo<CL_PLATFORM_NAME>();
        
        cout << "Using platform: " << platformName << endl;

        if (device == 0)
            default_platform.getDevices(CL_DEVICE_TYPE_CPU, &cl_devices);
        else
            default_platform.getDevices(CL_DEVICE_TYPE_GPU, &cl_devices);
        
        if (cl_devices.size() == 0) {
            cout << " No devices found. Check OpenCL installation!" << endl;
            exit(1);
        }

        cl::Device default_device = cl_devices[0];
        
        cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << endl;

        string kernelName = "updatephi.cl";
        char* clFile = oclLoadProgSource(kernelName.c_str(), "", &kernels_size);
//        cout << "cl File: " << clFile << endl;
        
        cl_sources = cl::Program::Sources(1, make_pair(clFile, kernels_size));

        cl_program = cl::Program(cl_context, cl_sources);

        cl_program.build(cl_devices, "-cl-fast-relaxed-math");

        cl_cmdqueue = cl::CommandQueue(cl_context, cl_devices[0]);
        
        cl_kernel = cl::Kernel(cl_program, "update_phi");

        cl::Buffer d_D = cl::Buffer(cl_context, CL_MEM_READ_WRITE, sizeof(float)*imageW*imageH);
        cl::Buffer d_phi = cl::Buffer(cl_context, CL_MEM_READ_WRITE, sizeof(float)*imageW*imageH);
        cl::Buffer d_phi1 = cl::Buffer(cl_context, CL_MEM_READ_WRITE, sizeof(float)*imageW*imageH);
        
    //    cl::ImageFormat imgFormat;
    //    imgFormat.image_channel_order = CL_R;
    //    imgFormat.image_channel_data_type = CL_UNORM_INT8;
    //    size_t origin[3] = {0, 0, 0};
    //    size_t region[3] = {imageW, imageH, 1};
        
    //    cl::Image2D imD = cl::Image2D(cl_context, CL_MEM_READ_WRITE, imgFormat, imageW, imageH, 0, NULL);
    //    cl::Image2D imPhi = cl::Image2D(cl_context, CL_MEM_READ_WRITE, imgFormat, imageW, imageH, 0, NULL);
    //    cl::Image2D imPhi1 = cl::Image2D(cl_context, CL_MEM_READ_WRITE, imgFormat, imageW, imageH, 0, NULL);

        cl_cmdqueue.enqueueWriteBuffer(d_D, CL_TRUE, 0, sizeof(float)*imageW*imageH, D, NULL, NULL);
        cl_cmdqueue.enqueueWriteBuffer(d_phi1, CL_TRUE, 0, sizeof(float)*imageW*imageH, phi, NULL, NULL);
        
    //    cl_cmdqueue.enqueueWriteImage(imD, CL_TRUE, origin, region, 0, 0, D, NULL, NULL);
    //    cl_cmdqueue.enqueueWriteImage(imD, CL_TRUE, origin, region, 0, 0, phi, NULL, NULL);
        
        int tsx = 16;
        int tsy = 16;
        
        cl_kernel.setArg(0, d_phi);
        cl_kernel.setArg(1, d_phi1);
        cl_kernel.setArg(2, d_D);
        cl_kernel.setArg(3, imageW);
        cl_kernel.setArg(4, imageH);
//        cl_kernel.setArg(5, tsx);
//        cl_kernel.setArg(6, tsy);
        
//        cl_kernel.setArg(0, imPhi);
//        cl_kernel.setArg(1, imPhi1);
//        cl_kernel.setArg(2, imD);
//        cl_kernel.setArg(3, imageW);
//        cl_kernel.setArg(4, imageH);
//        cl_kernel.setArg(5, tsx);
//        cl_kernel.setArg(6, tsy);
        
        double start = omp_get_wtime();

//        cout << "Width: " << imageW << endl;
//        cout << "Height: " << imageH << endl;
        cout << "WxH: " << N << endl;
        
        for (its=0; its<=ITERATIONS; its++)
        {
            // dim3 dimGrid( ((imageW-1)/BLOCKDIM_X) + 1, ((imageH-1)/BLOCKDIM_Y) +1 );
            // dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y);

            double start1 = omp_get_wtime();
//            cout << "before enqueue" << endl;
            
            cl_cmdqueue.enqueueNDRangeKernel(cl_kernel,
                                         cl::NullRange, cl::NDRange(imageW, imageH),
                                         cl::NullRange, NULL, &cl_event);
            
//            cl_cmdqueue.enqueueNDRangeKernel(cl_kernel,
//                                             cl::NullRange, cl::NDRange(imageW/tsy, imageH/tsx),
//                                             cl::NullRange, NULL, &cl_event);
            
            cl_event.wait();
            
            double end1 = omp_get_wtime();
//            cout << "Runtime i: " << its << ", " << end1 - start1 << endl;

            cl_kernel.setArg(0, d_phi1);
//            cl_kernel.setArg(0, imPhi1);

    //		if (its % 50 == 0)
    //			printf("Iteration %3d Total Time: %3.5f\n", its, 0.001*(end1 - start));
        }

        double end = omp_get_wtime();
        cout << "Total Runtime: " << end - start << endl;

        cl_cmdqueue.enqueueReadBuffer(d_phi1, CL_TRUE, 0, sizeof(float)*imageW*imageH, phi, NULL, NULL);
    //    cl_cmdqueue.enqueueReadBuffer(imPhi1, CL_TRUE, 0, sizeof(float)*imageW*imageH, phi, NULL, NULL);

    } catch (cl::Error cl_err) {
        printf("--- ERROR --- Main --- %s (%s)\n", cl_err.what(), oclErrorString(cl_err.err()));
        
        if (cl_err.err() == CL_BUILD_PROGRAM_FAILURE) {
            cl::STRING_CLASS buildLog;
            cl_program.getBuildInfo(cl_devices[device], CL_PROGRAM_BUILD_LOG, &buildLog);
            cout << buildLog << endl;
        }
        exit(1);
    }
        
	unsigned char *output;
	if ((output = (unsigned char *)malloc(N)) == NULL)
        printf("ME_OUTPUT\n");

	for (i=0; i<N; i++)
    {
		if (phi[i] > 0)
			output[i] = 0;
		else
            output[i] = 255;
	}

	char *outputFilename = "output.raw";
	FILE *fp = fopen(outputFilename, "wb");
	size_t write = fwrite(output, 1, N, fp);
	fclose(fp);
    printf("Write '%s', %d bytes\n", outputFilename, write);

//	char dummy[100];
//	scanf("%c",dummy);
}