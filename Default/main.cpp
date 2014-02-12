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

int main(int argc, char** argv) {
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
  /* Initialize with OpenCL kernel source file. If this changes, change here */
  se.initialize(std::string("silhouette_extraction.cl"));
  se.setDebug(debug_lvl);
  se.performTrainingStep();
  while(true) {
    if(!se.extractFromNextFrame()) {
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
     // cvWaitKey(5);
     // se.showOutput();
  }
   return 0;
}
