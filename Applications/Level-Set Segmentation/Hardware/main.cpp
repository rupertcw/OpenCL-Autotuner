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
#include <fstream>
#include <OpenCL/cl.hpp>

#include "../CL/oclUtils.h"

using namespace std;

const char* IMAGE = "input.bmp";
const int ITERATIONS = 1; //5000;
const int THRESHOLD = 180;
const int EPSILON = 45;

const char* NVIDIA = "NVIDIA";
const int NVIDIA_STEP = 32;

const char* AMD = "AMD";
const int AMD_STEP = 16;

const char* INTEL = "Intel";
const int INTEL_STEP = 8;

typedef struct {
    unsigned char x, y, z, w;
} uchar4;

uchar4 *h_Src, *h_Mask;

int imageW, imageH, N, pitch;
int its = 0;
int r, c, i;
int step = 1;

float *phi, *D;
float *d_phi, *d_D;
float *d_phi1;

void LoadBMPFile(uchar4 **dst, int *width, int *height, const char *name);
void sedt2d(int *_d,unsigned char *_bimg,int _h,int _w);

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
    
    ofstream myfile;
    myfile.open ("results.txt");
    
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
    
    int tsx, tsy, workgroupSize1 = 0, workgroupSize2 = 0, bestTx, bestTy, bestWorkgroupSize1, bestWorkgroupSize2;
    
    double bestTime = 999999.0;
    
    try {
        cl::Platform::get(&cl_platforms);

        if (cl_platforms.size() == 0) {
            printf("--- ERROR --- OpenCL platform-size is 0. Exiting.\n");
            exit(1);
        }

        cl_context_properties cl_properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(cl_platforms[0])(), 0};

//        if (device == 0)
//            cl_context = cl::Context(CL_DEVICE_TYPE_CPU, cl_properties);
//        else
            cl_context = cl::Context(CL_DEVICE_TYPE_GPU, cl_properties);

        cl_devices = cl_context.getInfo<CL_CONTEXT_DEVICES>();

        vector<cl::Platform> all_platforms;
        cl::Platform::get(&all_platforms);
        
        if (all_platforms.size() == 0) {
            cout << " No platforms found. Check OpenCL installation!" << endl;
            exit(1);
        }
        
        cl::Platform default_platform = all_platforms[0];
        string platformName = default_platform.getInfo<CL_PLATFORM_NAME>();
        cout << "Platform name: " << platformName << endl;

//        if (device == 0)
//            default_platform.getDevices(CL_DEVICE_TYPE_CPU, &cl_devices);
//        else
            default_platform.getDevices(CL_DEVICE_TYPE_GPU, &cl_devices);
        
        if (cl_devices.size() == 0) {
            cout << " No devices found. Check OpenCL installation!" << endl;
            exit(1);
        }

        cl::Device default_device = cl_devices[0];
        string deviceName = default_device.getInfo<CL_DEVICE_NAME>();
        string deviceVendor = default_device.getInfo<CL_DEVICE_VENDOR>();
        
        cout << "Vendor: " << deviceVendor << endl;
        cout << "Using device: " << deviceName << endl;
        
        if (deviceVendor.compare(0, strlen(NVIDIA), NVIDIA) == 0) {
            cout << "Stepping: NVIDIA" << endl;
            step = NVIDIA_STEP;
            workgroupSize1 += step;
            workgroupSize2 += step;
        } else if (deviceVendor.compare(0, strlen(AMD), AMD) == 0) {
            cout << "Stepping: AMD" << endl;
            step = AMD_STEP;
            workgroupSize1 += step;
            workgroupSize2 += step;
        } else if ((deviceVendor.compare(0, strlen(INTEL), INTEL) == 0) && (device == 1)) {
            cout << "Stepping: INTEL" << endl;
            step = INTEL_STEP;
            workgroupSize1 += step;
            workgroupSize2 += step;
        } else {
            cout << "Stepping: Normal" << endl;
            step = 1;
            workgroupSize1 += 1;
            workgroupSize2 += 1;
        }

        string kernelName = "updatephi.cl";
        char* clFile = oclLoadProgSource(kernelName.c_str(), "", &kernels_size);
        
        cl_sources = cl::Program::Sources(1, make_pair(clFile, kernels_size));

        cl_program = cl::Program(cl_context, cl_sources);

        cl_program.build(cl_devices, "-cl-fast-relaxed-math");

        cl_cmdqueue = cl::CommandQueue(cl_context, cl_devices[0]);
        
        cl_kernel = cl::Kernel(cl_program, "update_phi");

        cl::Buffer d_D = cl::Buffer(cl_context, CL_MEM_READ_WRITE, sizeof(float)*imageW*imageH);
        cl::Buffer d_phi = cl::Buffer(cl_context, CL_MEM_READ_WRITE, sizeof(float)*imageW*imageH);
        cl::Buffer d_phi1 = cl::Buffer(cl_context, CL_MEM_READ_WRITE, sizeof(float)*imageW*imageH);

        cl_cmdqueue.enqueueWriteBuffer(d_D, CL_TRUE, 0, sizeof(float)*imageW*imageH, D, NULL, NULL);
        cl_cmdqueue.enqueueWriteBuffer(d_phi1, CL_TRUE, 0, sizeof(float)*imageW*imageH, phi, NULL, NULL);

//        cout << "WxH: " << N << endl;
        int maxWorkgroupSize = (int)cl_kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(default_device);
        cout << "Kernel Max WGS: " << maxWorkgroupSize << endl;
        cout << endl;
        
        cl_kernel.setArg(3, imageW);
        cl_kernel.setArg(4, imageH);
        
        for (tsy = 1; tsy <= imageW; tsy++)
        {
            cl_kernel.setArg(6, tsy);
            
            for (tsx = 1; tsx <= imageH; tsx++)
            {
                cl_kernel.setArg(5, tsx);
                
                for (workgroupSize1 = step; workgroupSize1 <= maxWorkgroupSize; workgroupSize1 += step)
                {
                    int width = imageW;
                    
                    while ((width % tsx != 0) || ((width/tsx) % workgroupSize1 != 0))
                        width++;
                    
                    for (workgroupSize2 = step; workgroupSize2 <= maxWorkgroupSize; workgroupSize2 += step)
                    {
                        if ((workgroupSize1*workgroupSize2) > maxWorkgroupSize) {
                            cout << "Max workgroup size exceeded" << endl;
                            cout << "TS: " << tsx << ", " << tsy << endl;
                            cout << "LWGS: " << workgroupSize1 << ", " << workgroupSize2 << endl;
                            break;
                        }
                        
                        int height = imageH;
                        
                        while ((height % tsy != 0) || ((height/tsy) % workgroupSize2 != 0))
                            height++;
                        
                        cout << "Width, Height: " << width << ", " << height << endl;
                        cout << "TS: " << tsx << ", " << tsy << endl;
                        cout << "LWGS: " << workgroupSize1 << ", " << workgroupSize2 << endl;
                        
                        // RESET DATA
                        cl::Buffer d_phi = cl::Buffer(cl_context, CL_MEM_READ_WRITE, sizeof(float)*imageW*imageH);
                        cl_cmdqueue.enqueueWriteBuffer(d_D, CL_TRUE, 0, sizeof(float)*imageW*imageH, D, NULL, NULL);
                        cl_cmdqueue.enqueueWriteBuffer(d_phi1, CL_TRUE, 0, sizeof(float)*imageW*imageH, phi, NULL, NULL);
                        
                        cl_kernel.setArg(0, d_phi);
                        cl_kernel.setArg(1, d_phi1);
                        cl_kernel.setArg(2, d_D);
                        
                        double start = omp_get_wtime();
                        
                        for (its=0; its<=ITERATIONS; its++)
                        {
//                            double start1 = omp_get_wtime();
                            
                            cl_cmdqueue.enqueueNDRangeKernel(cl_kernel,
                                                             cl::NullRange, cl::NDRange(width/tsx, height/tsy),
                                                             cl::NDRange(workgroupSize1, workgroupSize2), NULL, &cl_event);
                            cl_event.wait();
                            
//                            double end1 = omp_get_wtime();

                            cl_kernel.setArg(0, d_phi1);
                        }
                        
                        double end = omp_get_wtime();
                        double newTime = end - start;
                        cout << "This Time: " << newTime << "s" << endl;
                        cout << endl;
                        
                        if (newTime < bestTime) {
                            bestTime = newTime;
                            bestTx = tsx;
                            bestTy = tsy;
                            bestWorkgroupSize1 = workgroupSize1;
                            bestWorkgroupSize2 = workgroupSize2;
                        }
                        
                        myfile << tsx;
                        myfile << ",";
                        myfile << tsy;
                        myfile << ",";
                        myfile << workgroupSize1;
                        myfile << ",";
                        myfile << workgroupSize2;
                        myfile << ",";
                        myfile << newTime;
                        myfile << "\n";
                    }
                }
            }
        }
        
        cout << "Best Time: " << bestTime << " - TS: (" << tsx << ", " << tsy << ") - WGS: (" << workgroupSize1 << ", " << workgroupSize2 << ")" << endl;

        cl_cmdqueue.enqueueReadBuffer(d_phi1, CL_TRUE, 0, sizeof(float)*imageW*imageH, phi, NULL, NULL);

    } catch (cl::Error cl_err) {
        printf("--- ERROR --- Main --- %s (%s)\n", cl_err.what(), oclErrorString(cl_err.err()));
        
        if (cl_err.err() == CL_BUILD_PROGRAM_FAILURE) {
            cl::STRING_CLASS buildLog;
            cl_program.getBuildInfo(cl_devices[device], CL_PROGRAM_BUILD_LOG, &buildLog);
            cout << buildLog << endl;
        }
    }
        
	unsigned char *output;
	if ((output = (unsigned char *)malloc(N)) == NULL)
        printf("ME_OUTPUT\n");
    
    myfile.close();

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