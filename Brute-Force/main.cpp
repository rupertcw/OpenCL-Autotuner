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

using namespace std;

int main(int argc, char** argv) {
    struct timeval start, end;
    int prevTime = 0, newTime = 0, bestTime = 9999999999, bestWorkgroupSize=0;
    
    if(argc <=3) {
        printf("Usage: ./SilhouetteExtraction filename num_training_frames debug_level\n");
        exit(1);
    }
    std::string filename(argv[1]);
    int num_training_frames = atoi(argv[2]);
    int debug_lvl = atoi(argv[3]);

    if(debug_lvl >= 1) {
        printf("--- MAIN --- Starting up.\n");
        printf("--- MAIN --- Using file \"%s\" with %d training-frames.\n",filename.c_str(),num_training_frames);
    }

    SilhouetteExtraction se(filename, num_training_frames, true);
    int maxWorkgroupSize = se.getMaxWorkGroupSize(string("silhouette_extraction.cl"));
//    cout << "in main: " << maxWorkGroupSize << endl;
    if (maxWorkgroupSize != 0) {
        for (int workgroupSize=3; workgroupSize <= maxWorkgroupSize; workgroupSize++) {
            cout << "Workgroup size: " << workgroupSize << endl;
            gettimeofday(&start, NULL);
            
            SilhouetteExtraction se(filename, num_training_frames, true);
            /* Initialize with OpenCL kernel source file. If this changes, change here */
            se.initialize(string("silhouette_extraction.cl"));
            se.setDebug(debug_lvl);
            
            bool ret = se.performTrainingStep(workgroupSize);
            while(true && ret == true) {
                if(!se.extractFromNextFrame(workgroupSize)) {
                    printf("--- MAIN --- Last frame found or error occured. Exiting.\n");
                    printf("--- MAIN --- Frames read in this run: %d.\n",se.getFrameCount());
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
        //        cvWaitKey(5);
        //        se.showOutput();
            }
            gettimeofday(&end, NULL);
            newTime = (int)((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec);
            
            if (ret && (newTime < bestTime)){
                bestTime = newTime;
                bestWorkgroupSize = workgroupSize;
            }
            prevTime = newTime;
            cout << "This Time: " << newTime << endl;
            cout << "Best Time: " << bestTime << endl;
        }
    }
    
    cout << "Top performance - Runtime: " << bestTime << ", Workgroup Size: " << bestWorkgroupSize << endl;
    
    return 0;
}
