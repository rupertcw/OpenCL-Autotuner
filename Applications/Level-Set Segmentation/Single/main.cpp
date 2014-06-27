#define __CL_ENABLE_EXCEPTIONS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <sstream>
#include <string>
#include <fstream>

#include <OpenCL/cl.hpp>

#include "../CL/oclUtils.h"
#include "Individual.h"

using namespace std;

typedef struct {
    unsigned char x, y, z, w;
} uchar4;

const char* IMAGE = "input.bmp";
const int ITERATIONS = 5; //5000;
const int THRESHOLD = 180;
const int EPSILON = 45;

const int N = 20;
const int nGenerations = 25;
const int nParams = 4;

ofstream myfile;

int its = 0;
int r, c, i;
int imageW, imageH, size;
int device, maxWorkgroupSize;

float *phi, *D;
float *d_phi, *d_D;
float *d_phi1;

uchar4 *h_Src, *h_Mask;

default_random_engine generator;

uniform_real_distribution<float> distribution(0.0, 1.0);

void LoadBMPFile(uchar4 **dst, int *width, int *height, const char *name);
void sedt2d(int *_d,unsigned char *_bimg,int _h,int _w);

void printPopulation(vector<Individual> P)
{
	for (int i=0; i<P.size(); i++)
		cout << "Individual " << i << ": " << P[i] << endl;
}

void printNormative(vector<vector<int> > normative)
{
	for (int i=0; i<normative.size(); i++)
		cout << "[" << normative[i][0] << ", " << normative[i][1] << "]" << endl;
}

int randint(int min, int max) { return rand() % max + min; }

// NOT RANDOM ENOUGH
float uniform() { return distribution(generator); }

void extend(vector<Individual>& v1, vector<Individual> v2)
{
	for (int i = 0; i < v2.size(); i++)
		v1.push_back(v2[i]);
}

bool fitnessComparison(Individual p, Individual q)
{
	if (p.fitness < q.fitness)
		return true;
	else
		return false;
}

float obj1(Individual individual)
{
    size_t kernels_size;
    vector<cl::Platform> cl_platforms;
    vector<cl::Device> cl_devices;
    cl::Program cl_program;
    cl::Context cl_context;
    cl::Program::Sources cl_sources;
    cl::CommandQueue cl_cmdqueue;
    cl::Event cl_event;
    cl::Kernel cl_kernel;
    
    int tsx = individual.values[1], tsy = individual.values[0], workgroupSize1 = individual.values[2], workgroupSize2 = individual.values[3];
    
    float eTime = 999999.0f;
    
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
        
        vector<cl::Platform> all_platforms;
        cl::Platform::get(&all_platforms);
        
        if (all_platforms.size() == 0) {
            cout << " No platforms found. Check OpenCL installation!" << endl;
            exit(1);
        }
        
        cl::Platform default_platform = all_platforms[0];
        string platformName = default_platform.getInfo<CL_PLATFORM_NAME>();
        
//        cout << "Using platform: " << platformName << endl;
        
        if (device == 0)
            default_platform.getDevices(CL_DEVICE_TYPE_CPU, &cl_devices);
        else
            default_platform.getDevices(CL_DEVICE_TYPE_GPU, &cl_devices);
        
        if (cl_devices.size() == 0) {
            cout << " No devices found. Check OpenCL installation!" << endl;
            exit(1);
        }
        
        cl::Device default_device = cl_devices[0];
        
//        cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << endl;
        
        string kernelName = "updatephi.cl";
        char* clFile = oclLoadProgSource(kernelName.c_str(), "", &kernels_size);
        
        cl_sources = cl::Program::Sources(1, make_pair(clFile, kernels_size));
        
        cl_program = cl::Program(cl_context, cl_sources);
        
        cl_program.build(cl_devices, "-cl-fast-relaxed-math");
        
        cl_cmdqueue = cl::CommandQueue(cl_context, cl_devices[0]);
        
        cl_kernel = cl::Kernel(cl_program, "update_phi");
        
        cl::Buffer d_D = cl::Buffer(cl_context, CL_MEM_READ_WRITE, sizeof(float)*imageW*imageH);
        cl::Buffer d_phi1 = cl::Buffer(cl_context, CL_MEM_READ_WRITE, sizeof(float)*imageW*imageH);
        
        cl::Buffer d_phi = cl::Buffer(cl_context, CL_MEM_READ_WRITE, sizeof(float)*imageW*imageH);
        cl_cmdqueue.enqueueWriteBuffer(d_D, CL_TRUE, 0, sizeof(float)*imageW*imageH, D, NULL, NULL);
        cl_cmdqueue.enqueueWriteBuffer(d_phi1, CL_TRUE, 0, sizeof(float)*imageW*imageH, phi, NULL, NULL);
        
//        cout << "WxH: " << size << endl;
        
        cl_kernel.setArg(0, d_phi);
        cl_kernel.setArg(1, d_phi1);
        cl_kernel.setArg(2, d_D);
        cl_kernel.setArg(3, imageW);
        cl_kernel.setArg(4, imageH);
        cl_kernel.setArg(5, tsx);
        cl_kernel.setArg(6, tsy);
        
        int width = imageW;
        int height = imageH;
        
        while ((width%tsy != 0) || (width%workgroupSize1 != 0))
            width++;
        
        while ((height%tsx != 0) || (height%workgroupSize2 != 0))
            height++;
        
        if ((workgroupSize1*workgroupSize2) > maxWorkgroupSize) {
            cout << "Max workgroup size exceeded" << endl;
            cout << endl;
            return 999999.0f;
        }
        
        cout << "W, H: " << width << ", " << height << endl;
        cout << "TS1, TS2: " << tsy << ", " << tsx << endl;
        cout << "LWGS1, LWGS2: " << workgroupSize1 << ", " << workgroupSize2 << endl;
        
        double start = omp_get_wtime();
        
        for (its=0; its<=ITERATIONS; its++)
        {
            double start1 = omp_get_wtime();
            
//            cl_cmdqueue.enqueueNDRangeKernel(cl_kernel,
//                                             cl::NullRange, cl::NDRange(width/tsy, height/tsx),
//                                             cl::NullRange, NULL, &cl_event);
            
//            cl_cmdqueue.enqueueNDRangeKernel(cl_kernel,
//                                             cl::NullRange, cl::NDRange(width/tsx, height/tsy),
//                                             cl::NDRange(workgroupSize1, workgroupSize2), NULL, &cl_event);
            cl_cmdqueue.enqueueNDRangeKernel(cl_kernel,
                                             cl::NullRange, cl::NDRange(width/tsx, height/tsy),
                                             cl::NullRange, NULL, &cl_event);
            
            cl_event.wait();
            
            double end1 = omp_get_wtime();
//            cout << "Runtime i: " << its << ", " << end1 - start1 << endl;
            
            cl_kernel.setArg(0, d_phi1);
            
            //		if (its % 50 == 0)
            //			printf("Iteration %3d Total Time: %3.5f\n", its, 0.001*(end1 - start));
        }
        
        double end = omp_get_wtime();
        eTime = (float)(end - start);
        cout << "Time: " << eTime << "s" << endl;
        cout << endl;
        
        myfile << tsx;
        myfile << ",";
        myfile << tsy;
        myfile << ",";
        myfile << workgroupSize1;
        myfile << ",";
        myfile << workgroupSize2;
        myfile << ",";
        myfile << eTime;
        myfile << "\n";
        
        cl_cmdqueue.enqueueReadBuffer(d_phi1, CL_TRUE, 0, sizeof(float)*imageW*imageH, phi, NULL, NULL);
        
    } catch (cl::Error cl_err) {
        printf("--- ERROR --- Main --- %s (%s)\n", cl_err.what(), oclErrorString(cl_err.err()));
        
        if (cl_err.err() == CL_BUILD_PROGRAM_FAILURE) {
            cl::STRING_CLASS buildLog;
            cl_program.getBuildInfo(cl_devices[device], CL_PROGRAM_BUILD_LOG, &buildLog);
            cout << buildLog << endl;
        }
//        exit(1);
    }
    
    return eTime;
}

void initialisePopulation(vector<Individual>& P, vector<vector<int> > searchSpace)
{
	int i = 0;
    
	while (i < N)
	{
		vector<int> params(nParams);
        
		for (int j=0; j<params.size(); j++)
			params[j] = randint(searchSpace[j][0], searchSpace[j][1]);
        
        if (!((params[2]*params[3]) > maxWorkgroupSize)) {
            P.push_back(Individual(params));
            i = i + 1;
        }
	}
    
	random_shuffle(P.begin(), P.end());
}

void initialiseBeliefSpace(vector<vector<int> > searchSpace, Individual& situational, vector<vector<int> >& normative)
{
	situational = Individual();
	normative = searchSpace;
}

void evaluatePopulation(vector<Individual>& P)
{
	for (int i=0; i<P.size(); i++)
		P[i].fitness = obj1(P[i]);
}

void sortPopulation(vector<Individual>& P, int co)
{
	sort(P.begin(), P.end(), fitnessComparison);
    
	if (co == 1)
		reverse(P.begin(), P.end());
}

void updateSituational(Individual& situational, Individual best)
{
	if ((situational.values.empty()) || (best.fitness < situational.fitness))
		situational = best;
}

float randInBounds(int min, int max)
{
	return min + ((max - min)* uniform());
}

void mutate(Individual individual, vector<int>& params, vector<vector<int> > normative, vector<vector<int> > searchSpace)
{
	params.clear();
    
	for (int i=0; i<individual.values.size(); i++)
	{
		float value = randInBounds(normative[i][0], normative[i][1]);
        
		if (value < searchSpace[i][0])
			value = searchSpace[i][0];
		if (value > searchSpace[i][1])
			value = searchSpace[i][1];
        
		params.push_back((int)value);
	}
}

void newGeneration(vector<Individual>& P, vector<Individual>& G, vector<vector<int> > normative, vector<vector<int> > searchSpace)
{
	vector<int> params;
    
	G.clear();
    
	for (int i=0; i<P.size(); i++){
		mutate(P[i], params, normative, searchSpace);
        
		G.push_back(Individual(params));
	}
}

Individual binaryTournament(vector<Individual> P)
{
	int r1 = randint(0, P.size()-1);
	int r2 = randint(0, P.size()-1);
    
	while (r2 == r1)
		r2 = randint(0, P.size()-1);
    
	if (P[r1].fitness < P[r2].fitness)
		return P[r1];
	else
		return P[r2];
}

void selectNewPopulation(vector<Individual>& P, vector<Individual>& P2, vector<Individual> G)
{
	int i = 0;
    
	P2.clear();
	extend(P, G);
    
	while (i < (P.size()/2))
	{
		P2.push_back(binaryTournament(P));
		i = i + 1;
	}
}

void updateNormative(vector<vector<int> >& normative, vector<Individual> accepted)
{
	for (int i=0; i<nParams; i++)
	{
		vector<int> params;
        
		for (int j=0; j<accepted.size(); j++)
			params.push_back(accepted[j].values[i]);
        
		sort(params.begin(), params.end());
        
		normative[i][0] = params.front();
		normative[i][1] = params.back();
	}
}

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
    device = atoi(argv[1]);
    
    myfile.open ("results.txt");
    
    srand(time(0));
    
	int numAccepted = round(N*0.2);
    
    size_t kernels_size;
    
    vector<cl::Platform> cl_platforms;
    vector<cl::Device> cl_devices;
    
    cl::Program cl_program;
    cl::Context cl_context;
    cl::Program::Sources cl_sources;
    cl::CommandQueue cl_cmdqueue;
    cl::Event cl_event;
    cl::Kernel cl_kernel;
    
	Individual situational;
    
	vector<Individual> P;
	vector<Individual> P2;
	vector<Individual> G;
	vector<Individual> accepted;
    
	vector<int> ss1(2);
	vector<int> ss2(2);
	vector<int> ss3(2);
	vector<int> ss4(2);
    
	vector<vector<int> > searchSpace(4);
	vector<vector<int> > normative;
    
	// Load the Input Image using BMPLoader
	const char *image_path = IMAGE;
	LoadBMPFile(&h_Src, &imageW, &imageH, image_path);
	D = (float *)malloc(imageW*imageH*sizeof(float));
    
	for (r=0; r<imageH; r++)
		for (c=0; c<imageW; c++)
			D[r*imageW + c] = h_Src[r*imageW + c].x;

	size = imageW*imageH;

	// Threshold based on hash defined paramters
	for(i=0; i<size; i++)
		D[i] = EPSILON - fabs(D[i] - THRESHOLD);

	// Init phi to SDF
	init_phi();
    
    //////////////////////////////////////////////////
    
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
        
        vector<cl::Platform> all_platforms;
        cl::Platform::get(&all_platforms);
        
        if (all_platforms.size() == 0) {
            cout << " No platforms found. Check OpenCL installation!" << endl;
            exit(1);
        }
        
        cl::Platform default_platform = all_platforms[0];
        
        if (device == 0)
            default_platform.getDevices(CL_DEVICE_TYPE_CPU, &cl_devices);
        else
            default_platform.getDevices(CL_DEVICE_TYPE_GPU, &cl_devices);
        
        if (cl_devices.size() == 0) {
            cout << " No devices found. Check OpenCL installation!" << endl;
            exit(1);
        }
        
        cl::Device default_device = cl_devices[device];
        
        string kernelName = "updatephi.cl";
        char* clFile = oclLoadProgSource(kernelName.c_str(), "", &kernels_size);
        
        cl_sources = cl::Program::Sources(1, make_pair(clFile, kernels_size));
        
        cl_program = cl::Program(cl_context, cl_sources);
        
        cl_program.build(cl_devices, "-cl-fast-relaxed-math");
        
        cl_cmdqueue = cl::CommandQueue(cl_context, cl_devices[0]);
        
        cl_kernel = cl::Kernel(cl_program, "update_phi");

        maxWorkgroupSize = (int)cl_kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(default_device);
    } catch (cl::Error cl_err) {
        printf("--- ERROR --- Main --- %s (%s)\n", cl_err.what(), oclErrorString(cl_err.err()));
        
        if (cl_err.err() == CL_BUILD_PROGRAM_FAILURE) {
            cl::STRING_CLASS buildLog;
            cl_program.getBuildInfo(cl_devices[device], CL_PROGRAM_BUILD_LOG, &buildLog);
            cout << buildLog << endl;
        }
    }
    
	ss1[0] = 1;
	ss1[1] = imageW;
	ss2[0] = 1;
	ss2[1] = imageH;
	ss3[0] = 1;
	ss3[1] = maxWorkgroupSize;
	ss4[0] = 1;
	ss4[1] = maxWorkgroupSize;
	searchSpace[0] = ss1;
	searchSpace[1] = ss2;
	searchSpace[2] = ss3;
	searchSpace[3] = ss4;

    printNormative(searchSpace);
    
	initialisePopulation(P, searchSpace);
	printPopulation(P);
    
	initialiseBeliefSpace(searchSpace, situational, normative);
	printNormative(normative);
    
	evaluatePopulation(P);
	printPopulation(P);
    
	sortPopulation(P, 0);
	printPopulation(P);
    
	Individual best = P[0];
	cout << "Best: " << best << endl;
    
	updateSituational(situational, best);
	cout << "Updated situational: " << situational << endl;
    
    for (int i = 0; i < nGenerations; i++)
    {
        newGeneration(P, G, normative, searchSpace);
        cout << "New Generation" << endl;
        printPopulation(G);
        
        evaluatePopulation(G);
        cout << "After Evaluation" << endl;
        printPopulation(G);
        
        sortPopulation(G, 0);
        cout << "After Sort" << endl;
        printPopulation(G);
        
        best = G[0];
        cout << "Best: " << best << endl;
        
        updateSituational(situational, best);
        cout << "Updated situational: " << situational << endl;
        
        selectNewPopulation(P, P2, G);
        cout << "New Population" << endl;
        P = P2;
        printPopulation(P);
        
        sortPopulation(P, 1);
        cout << "After Sort" << endl;
        printPopulation(P);
        
        accepted.clear();
        for (int i = 0; i < numAccepted; i++)
            accepted.push_back(P[i]);
        cout << "Accepted Individuals" << endl;
        printPopulation(accepted);
        
        updateNormative(normative, accepted);
        cout << "Updated Normative" << endl;
        printNormative(normative);
    }
    
	cout << "Best Solution: " << situational << endl;
    
    //////////////////////////////////////////////////
        
	unsigned char *output;
	if ((output = (unsigned char *)malloc(size)) == NULL)
        printf("ME_OUTPUT\n");

	for (i=0; i<size; i++)
    {
		if (phi[i] > 0)
			output[i] = 0;
		else
            output[i] = 255;
	}

	char *outputFilename = "output.raw";
	FILE *fp = fopen(outputFilename, "wb");
	size_t write = fwrite(output, 1, size, fp);
	fclose(fp);
    printf("Write '%s', %d bytes\n", outputFilename, write);
    
    myfile.close();

//	char dummy[100];
//	scanf("%c",dummy);
}