/* Silhouette Extraction using GPU and OpenCL
 * Algorithm based on paper:
 * "Robust Foreground Extraction Technique Using Gaussian Family Model and Multiple Thresholds"
 * by Kim et. al.
 *
 * File:     main.cpp
 * Author:   Kristian Haga Karstensen <kristianhk@linux.com>
 * Contents: Library for silhouette extraction in video. 
 *           Test-program using the library.
 *
 * This software is released under the terms of the GNU Lesser
 * General Public License Version 2.1 (LGPLv2.1). See file COPYING.LESSER
 * for details.
 */
#include "Silhouette_Extraction.h"
#include <omp.h>

#define NVIDIA "NVIDIA"
#define NVIDIA_STEP 32
#define INTEL "INTEL"
#define INTEL_STEP 8
#define AMD "AMD"
#define AMD_STEP 16
#define BRUTE_FORCE_STEP 1


using namespace std;

int main(int argc, char** argv) {
//    struct timeval start, end;
//    time_t start, end;
    int prevTime = 0, bestWorkgroupSize=0, type = 0, step, workgroupSize = 0;
    double bestTime = numeric_limits<double>::max();
    string bruteForceFlag("-brute-force");
    string hwTargetedFlag("-hw-targeting");
    
    if (argc <= 3) {
        printf("Usage: ./SilhouetteExtraction filename num_training_frames debug_level [-brute-force, -hw-targeting]\n");
        exit(1);
    }
    
    string filename(argv[1]);
    int num_training_frames = atoi(argv[2]);
    int debug_lvl = atoi(argv[3]);
    string chosenFlag = argv[4];
    
    cout << chosenFlag << endl;
    
    if (debug_lvl >= 1) {
        printf("--- MAIN --- Starting up.\n");
        printf("--- MAIN --- Using file \"%s\" with %d training-frames.\n",filename.c_str(),num_training_frames);
    }
    
    SilhouetteExtraction se1(filename, num_training_frames, true, type);
    int maxWorkgroupSize = se1.getMaxWorkGroupSize(string("silhouette_extraction.cl"));
    
    if (chosenFlag.compare(bruteForceFlag) == 0) {
        cout << "--- MAIN --- Mode: Brute Force " << endl;
        step = BRUTE_FORCE_STEP;
        workgroupSize = 3;
        
    } else if (chosenFlag.compare(hwTargetedFlag) == 0) {
        cout << "--- MAIN --- Mode: H/W Targeting" << endl;
        
        if (se1.platformName.compare(0, strlen(NVIDIA), NVIDIA) == 0) {
            cout << "--- MAIN --- Stepping: NVIDIA" << endl;
            step = NVIDIA_STEP;
            workgroupSize += step;
        } else if (se1.platformName.compare(0, strlen(INTEL), INTEL) == 0) {
            cout << "--- MAIN --- Stepping: INTEL" << endl;
            step = INTEL_STEP;
            workgroupSize += step;
        } else if (se1.platformName.compare(0, strlen(AMD), AMD) == 0) {
            cout << "--- MAIN --- Stepping: AMD" << endl;
            step = AMD_STEP;
            workgroupSize += step;
        }
    }
    
    se1.~SilhouetteExtraction();
    maxWorkgroupSize = workgroupSize;
//    cout << "in main: " << maxWorkGroupSize << endl;
    if (maxWorkgroupSize != 0) {
        for (type = 0; type < 3; type++) {
            for (workgroupSize; workgroupSize <= maxWorkgroupSize; workgroupSize += step) {
                cout << "Workgroup size: " << workgroupSize << endl;
//                cout << "Type: " << type << endl;
                
                if ((workgroupSize*workgroupSize) > maxWorkgroupSize) {
                    cout << "Max workgroup size exceeded" << endl;
                    exit(1);
                }
                
//                gettimeofday(&start, NULL);
//                time(&start);
                double start = omp_get_wtime();
                
                SilhouetteExtraction se(filename, num_training_frames, true, type);
                se.initialize(string("silhouette_extraction.cl"), workgroupSize);
                se.setDebug(debug_lvl);
                
//                cout << "here" << endl;
                
                bool ret = se.performTrainingStep(workgroupSize);
//                cout << "here1" << endl;
                
                while(true && ret == true) {
                    if(!se.extractFromNextFrame(workgroupSize)) {
                        printf("--- MAIN --- Last frame found or error occured. Exiting.\n");
                        printf("--- MAIN --- Frames read in this run: %d.\n", se.getFrameCount());
                        se.printStats();
                        break;
                    }
                    /* Uncomment if you want to show the output.
                     This requires changes in the Silhouette_Extraction.cpp code,
                     reading out the image/buffer from GPU memory in the
                     function where you want output, and modifying the
                     showOutput-function to display the image correctly.
                     For now there are parts commented out in some of the
                     functions where we read out data and put foreground pixels as
                     white and bg pixels as black in a buffer called outTest2. */
//                    cvWaitKey(5);
                    se.showOutput();
//                    cout << "here2" << endl;
                }
//                gettimeofday(&end, NULL);
//                time(&end);
//                newTime = (int)((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec);
//                double newTime = difftime (end, start);
                double end = omp_get_wtime();
                double newTime = end - start;
                
                if (ret && (newTime < bestTime)) {
                    bestTime = newTime;
                    bestWorkgroupSize = workgroupSize;
                }
                prevTime = newTime;
                cout << "This Time: " << newTime << endl;
                cout << "Best Time: " << bestTime << endl;
            }
        }
    }
    
    cout << "Top performance - Runtime: " << bestTime << ", Workgroup Size: " << bestWorkgroupSize << ", type: " << type << endl;
    
    return 0;
}
