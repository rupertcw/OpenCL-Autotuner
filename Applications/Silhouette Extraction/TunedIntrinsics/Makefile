# Silhouette Extraction using GPU and OpenCL
# Algorithm based on paper:
# "Robust Foreground Extraction Technique Using Gaussian Family Model and Multiple Thresholds"
# by Kim et. al.
#
# File:     Makefile
# Author:   Kristian Haga Karstensen <kristianhk@linux.om>
# Contents: Library for silhouette extraction in video.
#           Makefile
#
# This software is released under the terms of the GNU Lesser
# General Public License Version 2.1 (LGPLv2.1). See file COPYING.LESSER
# for details.
#

all:
	g++ -O3 -g -Wall -w -std=c99 `pkg-config --cflags opencv` -I../CL main.cpp ../CL/oclUtils.cpp ../CL/shrUtils.cpp ../CL/cmd_arg_reader.cpp Silhouette_Extraction.cpp -o SilhouetteExtraction  `pkg-config --libs opencv` -lOpenCL -lgomp
clean:
	rm -f SilhouetteExtraction *~ *.log

# To compile, we require oclUtils.cpp and oclUtils.h in /usr/include/CL (or change the path above)
# These have to be downlaoded separately / as part of OpenCL/NVIDIA SDK, as they are not redistributatble within our license
# They are used for reading opencl kernel files, and can be replaced by something home-written if you need that.
