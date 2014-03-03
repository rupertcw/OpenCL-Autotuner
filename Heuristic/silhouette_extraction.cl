/* Silhouette Extraction using GPU and OpenCL
 * Algorithm based on paper:
 * "Robust Foreground Extraction Technique Using Gaussian Family Model and Multiple Thresholds"
 * by Kim et. al.
 *
 * File:     silhouette_extraction.cl
 * Author:   Kristian Haga Karstensen <kristianhk@linux.com>
 * Contents: Library for silhouette extraction in video. OpenCL kernel code.
 *
 * This software is released under the terms of the GNU Lesser
 * General Public License Version 2.1 (LGPLv2.1). See file COPYING.LESSER
 * for details.
 */

/* Values for thresholding. Now manually set - adjust to new video data.
   Trial and error. */
/* Video 1 */
#define K1 140
#define K2 265
#define K3 280
#define KBG 200

/* Video 2 */
/* #define K1 1000 */
/* #define K2 4000 */
/* #define K3 30000 */
/* #define KBG 20000 */

#define sThresh 12 // TODO: set correct value here if used. Not used now.
#define MAX_VAL_UINT 99999999 // This should be set to the actual max size of uint. works for now.

/* For testing, remove if not needed */
#define BLOCK_WIDTH_PAD 18
#define BLOCK_HEIGHT_PAD 18
#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16


/* Functions for closing and opening */

/* Writes the outer regions of the work group to global memory,
   so all the workgroups can synchronize the changes afterwards */
void write_borders_to_global(global uchar* output_img, 
			     local uchar* l_img, 
			     uint width, uint height, 
			     uint x, uint y,
			     uint lx, uint ly, 
			     uint block_width, uint block_height) {
  uint block_width_pad = block_width+2;
  uint global_id = (y*width)+x;
  uint local_id = (ly*block_width_pad)+lx;
  /* If we're at one of the borders, save the value to output_img */
  if(lx == 1 || lx == block_width || ly == 1 || ly == block_height) {
    output_img[global_id] = l_img[local_id];
  }
}

/* Reads border regions into local memory from input_img */
/* If out of bounds, default_val is set. This is either 0
   or 255, depending on the context.
   TODO: Might be a bug in here. */
void read_border_regions(__global uchar* input_img,
			 __local uchar* l_img,
			 uint width, uint height, uint x, uint y, 
			 uint lx, uint ly, uint block_width,
			 uint block_height, uint default_val) {
  uint block_width_pad = block_width + 2;
  //uint block_height_pad = block_height + 2;
  //uint global_id = (y*width)+x;

  /* Read border regions */
  /* Left side */
  if(lx == 1) {
    /* Bounds check */
    if(x > 0) {
      l_img[(block_width_pad*ly)+(lx-1)] = input_img[(y*width)+(x-1)];
    } else {
      l_img[(block_width_pad*ly)+(lx-1)] = default_val;
    }
    /* Top left corner */
    if(ly == 1 && y > 0 && x > 0) {
      l_img[(block_width_pad*(ly-1))+(lx-1)] = input_img[((y-1)*width)+(x-1)];
    } else {
      l_img[(block_width_pad*(ly-1))+(lx-1)] = default_val;
    }
    /* Bottom left corner */
    if(ly == block_height && y < (height-1) && x > 0) {
      l_img[(block_width_pad*(ly+1))+(lx-1)] = input_img[((y+1)*width)+(x-1)];
    } else {
      l_img[(block_width_pad*(ly+1))+(lx-1)] = default_val;
    }
  }
  
  /* Right side  */
  if(lx == block_width) {
    /* Bounds check */
    if(x<(width-1)) {      
      l_img[(block_width_pad*ly)+(lx+1)] = input_img[(y*width)+(x+1)];
    } else {
      l_img[(block_width_pad*ly)+(lx+1)] = default_val;
    }
    /* Top right corner */
    if(ly == 1 && y > 0 && x < (width-1)) {
      l_img[(block_width_pad*(ly-1))+(lx+1)] = input_img[((y-1)*width)+(x+1)];
    } else {
      l_img[(block_width_pad*(ly-1))+(lx+1)] = default_val;
    }
    /* Bottom right corner */
    if(ly == block_height && y < (height-1) && x < (width-1)) {
      l_img[(block_width_pad*(ly+1))+(lx+1)] = input_img[((y+1)*width)+(x+1)];
    } else {
      l_img[(block_width_pad*(ly+1))+(lx+1)] = default_val;
    }
  }

  /* Top */
  if(ly == 1) {
    /* Bounds check */
    if(y > 0) {
      l_img[(block_width_pad*(ly-1))+lx] = input_img[((y-1)*width)+x];
    } else {
      l_img[(block_width_pad*(ly-1))+lx] = default_val;
    }
  }
  
  /* /\* Bottom *\/ */
  if(ly == block_height) {
    /* Bounds check */
    if(y<(height-1)) {
      l_img[(block_width_pad*(ly+1))+lx] = input_img[((y+1)*width)+x];
    }
    else {
      l_img[(block_width_pad*(ly+1))+lx] = default_val;
    }
  }
}
/* End read border regions */

/* Find max value */
uchar find_max_value(__local uchar* l_img, uint block_width_pad,
		     uint lx, uint ly) {

  /* Initialize to 0 initially */
  //uchar my_val, left_val, right_val, top_val, bottom_val, topleft_val, topright_val, bottomleft_val, bottomright_val = 0;
    
  /* my_val = l_img[ly*block_width_pad+lx]; */

  /* /\* Check boundaries for neigh. pixels *\/ */
  /* left_val = l_img[(ly*block_width_pad)+(lx-1)]; */
  /* right_val = l_img[(ly*block_width_pad)+(lx+1)]; */
  /* top_val = l_img[(ly-1)*block_width_pad+lx]; */
  /* topleft_val = l_img[((ly-1)*block_width_pad)+(lx-1)]; */
  /* topright_val = l_img[((ly-1)*block_width_pad)+(lx+1)]; */
  /* bottom_val = l_img[(block_width_pad*(ly+1))+lx]; */
  /* bottomleft_val = l_img[((ly+1)*block_width_pad)+(lx-1)]; */
  /* bottomright_val = l_img[((ly+1)*block_width_pad)+(lx+1)]; */

  /* uchar max_top = 0; */
  /* uchar max_center = 0; */
  /* uchar max_bottom = 0; */
  
  /* max_top = max(max(topleft_val,top_val),topright_val); */
  /* max_center = max(max(left_val,my_val),right_val); */
  /* max_bottom = max(max(bottomleft_val,bottom_val),bottomright_val); */
  
  /* uchar max_all; */
  /* max_all = max(max(max_top,max_center),max_bottom); */
  
  /* return max_all; */
  
  uchar max_top, max_center, max_bottom;

  /* Variables for more readable img-positions */
  uint top_left = (block_width_pad * (ly-1)) + (lx - 1);
  uint top = (block_width_pad * (ly-1)) + lx;
  uint top_right = (block_width_pad * (ly-1)) + (lx + 1);
  uint left = (block_width_pad * ly) + (lx - 1);
  uint me = (block_width_pad * ly) + lx;
  uint right = (block_width_pad * ly) + (lx + 1);
  uint bottom = (block_width_pad * (ly + 1)) + lx;
  uint bottom_left = (block_width_pad * (ly + 1)) + (lx - 1);
  uint bottom_right = (block_width_pad * (ly + 1)) + (lx + 1);

  max_top = max(max(l_img[top_left],l_img[top]),l_img[top_right]);
  max_center = max(max(l_img[left],l_img[me]),l_img[right]);
  max_bottom = max(max(l_img[bottom_left],l_img[bottom]),l_img[bottom_right]);
  
  uchar max_all = max(max(max_top,max_center),max_bottom);

  return max_all;
}

/* Find min value */
uchar find_min_value(__local uchar* l_img, uint block_width_pad,
		     uint lx, uint ly) {
  /* Initialize to 255 initially */
  //uchar my_val, left_val, right_val, top_val, bottom_val, topleft_val, topright_val, bottomleft_val, bottomright_val = 255;
    
  /* my_val = l_img[ly*block_width_pad+lx]; */

  /* /\* Check boundaries for neigh. pixels *\/ */
  /* left_val = l_img[(ly*block_width_pad)+(lx-1)]; */
  /* right_val = l_img[(ly*block_width_pad)+(lx+1)]; */
  /* top_val = l_img[(ly-1)*block_width_pad+lx]; */
  /* topleft_val = l_img[((ly-1)*block_width_pad)+(lx-1)]; */
  /* topright_val = l_img[((ly-1)*block_width_pad)+(lx+1)]; */
  /* bottom_val = l_img[(block_width_pad*(ly+1))+lx]; */
  /* bottomleft_val = l_img[((ly+1)*block_width_pad)+(lx-1)]; */
  /* bottomright_val = l_img[((ly+1)*block_width_pad)+(lx+1)]; */

  /* uchar min_top = 0; */
  /* uchar min_center = 0; */
  /* uchar min_bottom = 0; */
  
  /* min_top = min(min(topleft_val,top_val),topright_val); */
  /* min_center = min(min(left_val,my_val),right_val); */
  /* min_bottom = min(min(bottomleft_val,bottom_val),bottomright_val); */
  
  /* uchar min_all; */
  /* min_all = min(min(min_top,min_center),min_bottom); */
  
  /* return min_all; */

  uchar min_top, min_center, min_bottom;
  
  /* Variables for more readable img-positions */
  uint top_left = (block_width_pad * (ly-1)) + (lx - 1);
  uint top = (block_width_pad * (ly-1)) + lx;
  uint top_right = (block_width_pad * (ly-1)) + (lx + 1);
  uint left = (block_width_pad * ly) + (lx - 1);
  uint me = (block_width_pad * ly) + lx;
  uint right = (block_width_pad * ly) + (lx + 1);
  uint bottom = (block_width_pad * (ly + 1)) + lx;
  uint bottom_left = (block_width_pad * (ly + 1)) + (lx - 1);
  uint bottom_right = (block_width_pad * (ly + 1)) + (lx + 1);

  min_top = min(min(l_img[top_left],l_img[top]),l_img[top_right]);
  min_center = min(min(l_img[left],l_img[me]),l_img[right]);
  min_bottom = min(min(l_img[bottom_left],l_img[bottom]),l_img[bottom_right]);
  
  uchar min_all = min(min(min_top,min_center),min_bottom);

  return min_all;
}

/* End functions for closing and opening */

/* Functions for CCL */

/* findRoot - note __local uint - change if using smth else */
int findRoot(__local int *equivArray, int elemAddr) {
  int nextElem;
  while(elemAddr < nextElem) {
    nextElem = elemAddr;
    elemAddr = equivArray[nextElem];
  }
  return elemAddr;
}

int findRootGlobal(__global int *equivArray, int elemAddr) {
  int nextElem;
  while(elemAddr < nextElem) {
    nextElem = elemAddr;
    elemAddr = equivArray[nextElem];
  }
  return elemAddr;
}
// here
void unionF(global int* buf, uint seg1, uint seg2, int reg1, int reg2, local uint* changed) {
  if(seg1 == seg2)  {
    int newReg1 = findRootGlobal(buf, reg1);
    int newReg2 = findRootGlobal(buf, reg2);

    if(newReg1 > newReg2) {
      atom_min(buf+newReg1, newReg2);
      changed[0] = 1;
    }
    else if(newReg2 > newReg1) {
      atom_min(buf+newReg2, newReg1);
      changed[0] = 1;
    }
  }
}

/* Internal function for kernel 3 and 4 (CCL).
   Why log2width? It's not even used. */
void flattenEquivalenceTreesInternal(int x, int y, __global int* labelsOut, __global int* labelsIn, int width, int log2width)
{
  int index = x+y*width;
  int label = labelsIn[index];
  // flatten tree for all elements (non background)
  // where the elements labels are not roots of the equivalence tree
  if(label != index && label != -1) {
      int newLabel = findRootGlobal(labelsIn, label);
      if(newLabel < label) {
	// Set the label of the root elem to the label of the processed elem.
	labelsOut[index] = newLabel;
      }
  }		
}

/* TODO: other union function defined in example code (unionF).
   Need to figure that one out, probably have to use it */
/* uint cclUnion(__local uint* equivArray, uint elemAddr0, */
/* 	      uint elemAddr1) { */
/*   uint root0 = findRoot(equivArray,elemAddr0); */
/*   uint root1 = findRoot(equivArray,elemAddr1); */
  
/*   // Connect equiv tree with higher label with tree with a lower label */
/*   if(root0 < root1) */
/*     equivArray[root1] = root0; */
/*   if(root1 < root0) */
/*     equivArray[root0] = root1; */
/* } */

/* END Functions for CCL */

/* This kernel is used for conversion from RGB to luma (Y) and color (H) */
__kernel void convert_rgb_to_h_and_y(global uchar* rgbimg, global int2* Data, int width)
{
    unsigned int id = (get_global_id(1)*width) + get_global_id(0);
    int b = rgbimg[(id*3)]; // NOW BGR. TODO: what should we use? Seems OpenCV uses BGR
    int g = rgbimg[(id*3) + 1];
    int r = rgbimg[(id*3) + 2];

    int Y_val = (r*0.299) + (g*0.587) + (b*0.114);

    int H,S,I;
    H = 0;
    I = max(r,g);
    I = max(I,b);
        
    if (I==0) {
        S = 0;
    } else {
        int minRGB = min(r,g);
        minRGB = min(minRGB,b);
        S = (I - minRGB / I);
    }
        
    if (I == r) {
        H = (g-b) * 60;
        if (S != 0)
            H = H / S;
    } else if (I == g) {
        H = 180 + (b-r) * 60;
        if (S != 0)
            H = H / S;
    } else if (I == b) {
        H = 240 + (r-g) * 60;
        if (S != 0)
            H = H / S;
    }
        
    if(H < 0) {
        H = H + 360;
    }

    int2 converted_data;
    converted_data.s0 = Y_val;
    converted_data.s1 = H;

    /* Synchronize threads */

    barrier(CLK_GLOBAL_MEM_FENCE);

    /* Write back to global memory */
    Data[id] = converted_data;
}

/* Kernel for calculating mean, variance and excess kurtosis from the training data.
 * UPDATE: Now called for each new training frame, calculating the running mean and variance.
 * Based on Welford algorithm described in http://www.johndcook.com/standard_deviation.html.
 * Online/running kurtosis taken from Terriberry: http://people.xiph.org/~tterribe/notes/homs.html
 * described on Wikipedia: 
 * http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Higher-order_statistics 
 * (online_kurtosis)
 * N_max = total number of frames
 * size = image size (needed now? todo)
 * n = current frame number (start at 0)
 */
__kernel void calc_mean_variance_kurtosis(int N_max, int size, int n, global int2* data, global float4* M_V,
                    global float2* M_2, global float2* M_3, global float2* M_4, global int2* DistMap, int width)
{
    int id = (get_global_id(1)*width) + get_global_id(0);

    /* M_V contains: luma-mean, color-mean, luma-variance, color-variance, in that order */
    float4 MV = M_V[id];

    /* M2-4's have luma first, then color */
    float2 M2 = M_2[id];
    float2 M3 = M_3[id];
    float2 M4 = M_4[id];

    /* data: l first, then c */
    int2 lcData = data[id];
    
    /* Check if first run. In that case, initialize to zero.
     If not, keep values. Use step for this. first_step
     should become 0 if n < 1.0 */
    float first_step = 0;

    if(n == 0) {
        MV = MV * first_step; // multiply all elements
        M2 = M2 * first_step; // with 0.0 or 1.0
        M3 = M3 * first_step;
        M4 = M4 * first_step;
    }
    else
        first_step = 1.0;

    /* Calculate running mean, variance, excess kurtosis */
    /* Luma */
    int n1 = n;
    n = n+1;
    
    float delta = lcData.s0 - MV.s0;
    float delta_n = delta / n;
    float delta_n2 = delta_n * delta_n;
    float term1 = delta * delta_n * n1;
    
    // Update mean
    MV.s0 = MV.s0 + delta_n;
    
    // Update variance
    MV.s2 = MV.s2 + delta*(lcData.s0 - MV.s0);
    MV.s2 = MV.s2*first_step; // If first run, set to 0.0
    M4.s0 = M4.s0 + term1*delta_n2 * (n*n - 3*n + 3) + 6*delta_n2*M2.s0 - 4*delta_n*M3.s0;
    M3.s0 = M3.s0 + term1*delta_n*(n - 2) - 3*delta_n*M2.s0;
    M2.s0 = M2.s0 + term1;

    /* Color */
    delta = lcData.s1 - MV.s1;
    delta_n = delta / n;
    delta_n2 = delta_n*delta_n;
    term1 = delta*delta_n*n1;
    
    // Update mean
    MV.s1 = MV.s1 + delta_n;
    
    //Update variance
    MV.s3 = MV.s3 + delta * (lcData.s1 - MV.s1); 
    MV.s3 = MV.s3 * first_step; // If first run, set to 0.0
    M4.s1 = M4.s1 + term1 * delta_n2 * (n*n - 3*n + 3) + 6 * delta_n2 * M2.s0 - 4 * delta_n * M3.s1;
    M3.s1 = M3.s1 + term1 * delta_n * (n - 2) - 3 * delta_n * M2.s1;
    M2.s1 = M2.s1 + term1;

    /* Write back to memory */
    M_V[id] = MV;
    M_2[id] = M2;
    M_3[id] = M3;
    M_4[id] = M4;

    /* Last training frame? Calculate excess kurtosis 
     and choose distributions to use */
    if(n == N_max) {
        // Allocate int2 for distmap
        int2 lcDistMap;

        /* Calculate excess kurtosis: M4 is in .s3, M2 is in .s1
           (count*M4) / (M2*M2) - 3 */
        float lKurt = (n*M4.s0) / (M2.s0*M2.s0) - 3;
        float cKurt = (n*M4.s1) / (M2.s1*M2.s1) - 3;


        /* Set map for distribution to use */
        /* Using step-function to avoid the if's: */    
        // Will give 1+0 if 0.0 < lKurt (lKurt > 0.0)), 1+1 otherwise
        lcDistMap.s0 = (int) (step(lKurt, 0.0f) + 1);

        // Will give 1+0 if 0.0 < lKurt (lKurt > 0.0)), 1+1 otherwise
        lcDistMap.s1 = (int) (step(cKurt,0.0f) + 1);

        /* Save to DistMap in global mem */
        DistMap[id] = lcDistMap;
    }
}

/* Kernel for updating / creating the background model.
 Implemented as understood from the paper.
 Mean and variance gets updated for each run of the
 silhouette extraction (in the kernel at the bottom). */
__kernel void update_bg_model(global int2* Data, global float2* lc_Model, global float4* M_V, global int2* DistMap)
{
    /* id for this pixel */
    int id = get_global_id(1) * get_global_size(0) + get_global_id(0);

    /* M_V contains: luma-mean, color-mean,
     luma-variance, color-variance, in that order */
    float4 MV = M_V[id];
    float2 model = lc_Model[id];
    int2 dist_map = DistMap[id];
    int2 lc_data = Data[id];

    /* Standard deviation is sqrt of variance */
    float lStdDev = sqrt(MV.s2);
    float cStdDev = sqrt(MV.s3);
    int lDist = dist_map.s0;
    int cDist = dist_map.s1;

    /* Calculate luma model */
    float Yl = (1/lStdDev) * (pow((tgamma((float)3/lDist)) / (tgamma((float)1/lDist)), 0.5f));

    model.s0 = ((lDist * Yl) / (2*tgamma((float)1/lDist))) *
        exp(pow(-Yl, lDist) * pow((fabs((float)lc_data.s0 - MV.s0)), lDist));

    /* Calculate color model */
    float Yc = (1/cStdDev) * (pow((tgamma((float)3/cDist)) / (tgamma((float)1/cDist)), 0.5f));

    model.s1 = ((cDist * Yc) / (2*tgamma((float)1/cDist))) *
        exp(pow(-Yl, cDist) * pow((fabs((float)lc_data.s1 - MV.s1)), cDist));

    /* Write out model to global mem */
    lc_Model[id] = model;
}

/* Kernel for performing Background subtraction, thresholding image,
  performing morphological processes, connected components labeling etc.
 
  Done now: only BG subtraction and thresholding (thresholds are manually set,
  this is a weakness of this algorithm, don't have time to find a better way
  of doing this now. TODO: mention this in my thesis, try to suggest ways
  for automatic and dynamic thresholding that can be used to improve the algorithm.
 */
__kernel void subtract_and_threshold(global int2* Data, global float2* lcModel, global float4* M_V,
            __write_only image2d_t output_img, int width, global int* region_values,
            /* For closeAndOpenImage2 - remove after test */
            global uchar* output_buffer)
{
  int x = get_global_id(0);
  int y = get_global_id(1);

  int id = (y*width) + x;

  /* Get data */
  int2 lc_data = Data[id];
  float2 model = lcModel[id];
  float4 MV = M_V[id];

  float lData = (float) lc_data.s0;
  float cData = (float) lc_data.s1;
  float lModel = model.s0;
  float cModel = model.s1;

  /* Subtract luma data for current frame from luma model */
  float BD = fabs(lData-lModel);

  /* Initialize testresult to 0 */
  uint outval = 0;
  int region_val = 0;
  float lStdDev = sqrt(MV.s2);
  float cStdDev = sqrt(MV.s3);
  
  // TODO: use step here to get rid of ifs, if possible.

  // Background
  if(BD < (K1*lStdDev)) {
    outval = 0;
    region_val = 0;
  }
  // Suspicious background
  if((K1*lStdDev) <= BD && BD <= (K2*lStdDev)) {
    outval = 0;
    region_val = 1;
  }
  // Suspicious foreground
  if((K2*lStdDev) <= BD && BD <= (K3*lStdDev)) {
    outval = 1;
    region_val = 2;
    //    Eliminate shadows from suspicious foreground with help from color-data:
    if(fabs(cData - cModel) < (KBG*cStdDev)) {
      outval = 0;
      region_val = 1;
    }
  }	
  // Foreground
  if((K3*lStdDev) <= BD) {
    outval = 1;
    region_val = 3;
  }
  
  /* Write out thresholded mask to output_img */
  int2 coords = (int2)(x,y);
  uint4 output_mask;
  output_mask.s0 = outval; // TODO: this var-name says nothing. fix.

  write_imageui(output_img, ucoords, output_mask);
  region_values[id] = region_val;
  
  /* For closeAndOpenImage2 - TODO:  remove when done */
  output_buffer[id] = outval;
  /* End closeAndOpenImage2 */
}

/* Update mean and variance for background elements (running avg.)
   after refinement steps (close/open, ccl, removal of small elems, etc.).
   TODO: Expand this if needed (for counting and removing FG elems
   that have been stationary for <threshold> frames).
   input_img here is the segmented and processed image, 0 is background.
 */
__kernel void background_update(__read_only image2d_t input_img,
				__global int2* Data,
				__global float4* M_V) {
  int width = get_global_size(0);
  int x = get_global_id(0);
  int y = get_global_id(1);
  int id = (y*width) + x;
  float4 MV = M_V[id];

  const sampler_t sampl = CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP |
    CLK_FILTER_NEAREST;
  
  uint4 img_in = read_imageui(input_img, sampl, (int2)(x,y));

  int2 lc_data = Data[id];
  int lData = lc_data.s0;
  int cData = lc_data.s1;

  float oldLMean = MV.s0;
  float oldCMean = MV.s1;

  float Alpha = 0.0;
  if(img_in.s0 == 0) {
    Alpha = 0.05;
  }

  MV.s0 = (Alpha * lData) + (1.0-Alpha) * oldLMean;
  MV.s2 = Alpha * pow((lData-oldLMean),2) +
    (1-Alpha) * MV.s2;

  MV.s1 = (Alpha * cData) + (1.0-Alpha) * oldCMean;
  MV.s3 = Alpha * pow((cData-oldCMean),2) +
    (1-Alpha) * MV.s3;

  M_V[id] = MV;
}

/* Dilation 
   This implementation is assuming 
   a 3x3 structuring element. 
*/
__kernel void dilate_image(__read_only  image2d_t input_img,  __write_only image2d_t output_img) {
  /* Sampler for image */
  const sampler_t sampl = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  int x = get_global_id(0);
  int y = get_global_id(1);

  /* Get values, don't bother checking boundaries */
  uint4 my_val = read_imageui(input_img,sampl,(int2)(x, y));
  uint4 left_val = read_imageui(input_img,sampl,(int2)(x-1, y));
  uint4 right_val = read_imageui(input_img,sampl,(int2)(x+1, y));
  uint4 top_val = read_imageui(input_img,sampl,(int2)(x, y-1));
  uint4 bottom_val = read_imageui(input_img,sampl,(int2)(x, y+1));
  uint4 topleft_val = read_imageui(input_img,sampl,(int2)(x-1, y-1));
  uint4 topright_val = read_imageui(input_img,sampl,(int2)(x+1, y-1));
  uint4 bottomleft_val = read_imageui(input_img,sampl,(int2)(x-1, y+1));
  uint4 bottomright_val = read_imageui(input_img,sampl,(int2)(x+1, y+1));

  uint max_top = 0;
  uint max_center = 0;
  uint max_bottom = 0;
  
  max_top = max(max(topleft_val.s0,top_val.s0),topright_val.s0);
  max_center = max(max(left_val.s0,my_val.s0),right_val.s0);
  max_bottom = max(max(bottomleft_val.s0,bottom_val.s0),bottomright_val.s0);
  
  uint4 max_all;
  max_all.s0 = max(max(max_top,max_center),max_bottom);

  /* Write out */
  write_imageui(output_img,(int2)(x,y),max_all);
}

/* Opposite of dilate. */
__kernel void erode_image(__read_only  image2d_t input_img,  __write_only image2d_t output_img) {
   /* Sampler for image */
  const sampler_t sampl = CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP |
    CLK_FILTER_NEAREST;

  int x = get_global_id(0);
  int y = get_global_id(1);

  /* Get values, don't bother checking boundaries */
  uint4 my_val = read_imageui(input_img,sampl,(int2)(x, y));
  uint4 left_val = read_imageui(input_img,sampl,(int2)(x-1, y));
  uint4 right_val = read_imageui(input_img,sampl,(int2)(x+1, y));
  uint4 top_val = read_imageui(input_img,sampl,(int2)(x, y-1));
  uint4 bottom_val = read_imageui(input_img,sampl,(int2)(x, y+1));
  uint4 topleft_val = read_imageui(input_img,sampl,(int2)(x-1, y-1));
  uint4 topright_val = read_imageui(input_img,sampl,(int2)(x+1, y-1));
  uint4 bottomleft_val = read_imageui(input_img,sampl,(int2)(x-1, y+1));
  uint4 bottomright_val = read_imageui(input_img,sampl,(int2)(x+1, y+1));

  uint min_top = 0;
  uint min_center = 0;
  uint min_bottom = 0;
  
  min_top = min(min(topleft_val.s0,top_val.s0),topright_val.s0);
  min_center = min(min(left_val.s0,my_val.s0),right_val.s0);
  min_bottom = min(min(bottomleft_val.s0,bottom_val.s0),bottomright_val.s0);
  
  uint4 min_all;
  min_all.s0 = min(min(min_top,min_center),min_bottom);

  //  barrier(CLK_LOCAL_MEM_FENCE);
  
  /* Write out */
  write_imageui(output_img,(int2)(x,y), min_all);
}

/* START TEST WITH GLOBAL MEM */
/* Dilation 
   This implementation is assuming 
   a 3x3 structuring element. 
   This implementation uses global memory and boundary checking.
   Later we will compare this to using Image2D with built-in
   boundary checking and caching.
*/
__kernel void dilate_image_globalmem(global uchar* input_img, global uchar* output_img) {
  uint width = get_global_size(0);
  uint height = get_global_size(1);
  uint x = get_global_id(0);
  uint y = get_global_id(1);
  
  uint id = y*width+x;

  /* Initialize to 0 initially */
  uchar my_val, left_val, right_val, top_val,
    bottom_val, topleft_val, topright_val, 
    bottomleft_val, bottomright_val = 0;
    
  my_val = input_img[id];

  /* Check boundaries for neigh. pixels */
  if(x > 0) {
    left_val = input_img[(y*width)+(x-1)];
  }
  if(x < (width-1)) {
    right_val = input_img[(y*width)+(x+1)];
  }
  if(y > 0) {
    top_val = input_img[(y-1)*width+x];
    if(x > 0) {
      topleft_val = input_img[((y-1)*width)+(x-1)];
    }
    if(x < (width-1)) {
      topright_val = input_img[((y-1)*width)+(x+1)];
    }
  }
  if(y < (height-1)) {
    bottom_val = input_img[(width*(y+1))+x];
    if(x > 0) {
      bottomleft_val = input_img[((y+1)*width)+(x-1)];
    }
    if(x < (width-1)) {
      bottomright_val = input_img[((y+1)*width)+(x+1)];
    }
  }  

  uchar max_top = 0;
  uchar max_center = 0;
  uchar max_bottom = 0;
  
  max_top = max(max(topleft_val,top_val),topright_val);
  max_center = max(max(left_val,my_val),right_val);
  max_bottom = max(max(bottomleft_val,bottom_val),bottomright_val);
  
  uchar max_all;
  max_all = max(max(max_top,max_center),max_bottom);
    
  //barrier(CLK_GLOBAL_MEM_FENCE); needed?

  /* Write out */
  output_img[id] = max_all;
}
/* END TEST WITH GLOBAL MEM */

/* START TEST WITH GLOBAL MEM */
/* Opposite of dilate. */
__kernel void erode_image_globalmem(global uchar* input_img, global uchar* output_img) {

  uint width = get_global_size(0);
  uint height = get_global_size(1);
  uint x = get_global_id(0);
  uint y = get_global_id(1);
  
  uint id = y*width+x;

  /* Initialize to 0 initially */
  uchar my_val, left_val, right_val,
    top_val, bottom_val, topleft_val,
    topright_val, bottomleft_val, bottomright_val = 255;

  my_val = input_img[id];

  /* Check boundaries for neigh. pixels */
  if(x > 0) {
    left_val = input_img[(y*width)+(x-1)];
  }
  if(x < (width-1)) {
    right_val = input_img[(y*width)+(x+1)];
  }
  if(y > 0) {
    top_val = input_img[(y-1)*width+x];
    if(x > 0) {
      topleft_val = input_img[((y-1)*width)+(x-1)];
    }
    if(x < (width-1)) {
      topright_val = input_img[((y-1)*width)+(x+1)];
    }
  }
  if(y < (height-1)) {
    bottom_val = input_img[(width*(y+1))+x];
    if(x > 0) {
      bottomleft_val = input_img[((y+1)*width)+(x-1)];
    }
    if(x < (width-1)) {
      bottomright_val = input_img[((y+1)*width)+(x+1)];
    }
  }  

  uchar min_top = 0;
  uchar min_center = 0;
  uchar min_bottom = 0;
  
  min_top = min(min(topleft_val,top_val),topright_val);
  min_center = min(min(left_val,my_val),right_val);
  min_bottom = min(min(bottomleft_val,bottom_val),bottomright_val);
  
  uchar min_all;
  min_all = min(min(min_top,min_center),min_bottom);

  //  barrier(CLK_LOCAL_MEM_FENCE);
  
  /* Write out */
  output_img[id] = min_all;
}

/* END TEST WITH GLOBAL MEM */


/* Use this for writing out the image in 
   right format (for displaying, etc.).
   Not currently used - enqueueReadImage suddenly worked.*/
__kernel void writeout_imageui(__read_only image2d_t input_img, global uchar* out, int width) {
  const sampler_t sampl = CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP |
    CLK_FILTER_NEAREST;

  int x = get_global_id(0);
  int y = get_global_id(1);

  int id = (y*width)+x;

  int2 coords = (int2)(x,y);
  uint4 val = read_imageui(input_img, sampl, coords);

  out[id] = val.s0;
}

/* CCL algorithm. Kernel for initializing labels.
   Input is binary image result from closing/opening process.
   Output is a new image2d, with the labels set.
   Since we're only writing out labels here, and not reading
   and writing the same image, this will work fine.
   It gets a bit more complicated in the other functions,
   since we only can read OR write image2d_t in a kernel-run.
*/
__kernel void CCL_init_labels(__read_only image2d_t input_img, __write_only image2d_t output_img) {

  const sampler_t sampl = CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP |
    CLK_FILTER_NEAREST;
  
  int x = get_global_id(0);
  int y = get_global_id(1);
  int width = get_image_width(input_img);

  int2 coords = (int2)(x,y);
  uint4 val = read_imageui(input_img, sampl, coords);
  int pos = (y*width)+x;
  
  /* Value (1 or 0) * position-number */
  uint4 outval;
  outval.s0 = pos * val.s0;
  
  /* Write out */
  write_imageui(output_img, coords, outval);
}

/* Input here is output from CCL_init_labels */
__kernel void CCL_scanning(__read_only image2d_t input_img, __write_only image2d_t output_img,
			   global uint* isNotDone, global uint* img
			   /*, __local uint l_img[BLOCK_WIDTH_PAD*BLOCK_HEIGHT_PAD]*/) {
  const sampler_t sampl = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  int x = get_global_id(0);
  int y = get_global_id(1);
  int width = get_image_width(input_img);
  int pos = (y*width)+x;
  
  int2 my_c = (int2)(x,y);
  int2 left_c = (int2)(x-1, y);
  int2 right_c = (int2)(x+1, y);
  int2 top_c = (int2)(x, y-1);
  int2 bottom_c = (int2)(x, y+1);
  int2 topleft_c = (int2)(x-1,y-1);
  int2 topright_c = (int2)(x+1,y-1);
  int2 bottomleft_c = (int2)(x-1,y+1);
  int2 bottomright_c = (int2)(x+1,y+1);

  /* Get value for this thread's pixel */
  uint4 my_val = read_imageui(input_img, sampl, my_c);

  /* First, write this out to output_img */
  write_imageui(output_img, my_c, my_val);
  img[pos] = my_val.s0;

  if(x == 0 && y == 0) {
    isNotDone[0] = 0;
  }
  
  //  barrier(CLK_GLOBAL_MEM_FENCE);

  /* If pixel is not background: */
  if(my_val.s0) {
    //uint min_label = MAX_VAL_UINT; //TODO: fix this, now just defined something
    uint4 left_val4 = read_imageui(input_img,sampl,left_c);
    int zerostep = (int) step((float)left_val4.s0, 1.0f);
    zerostep *= MAX_VAL_UINT;
    left_val4.s0 += zerostep; // if zero, add MAX_VAL_UINT
    
    /* if(left_val.s0) { */
    /*   min_label = left_val.s0; */
    /* } */
    uint4 right_val4 = read_imageui(input_img,sampl,right_c);
    zerostep = (int) step((float)right_val4.s0, 1.0f);
    zerostep *= MAX_VAL_UINT;
    right_val4.s0 += zerostep;
    /* if(right_val4.s0 && right_val4.s0 < min_label) { */
    /*   min_label = right_val4.s0; */
    /* } */
    uint4 top_val4 = read_imageui(input_img,sampl,top_c);
    zerostep = (int) step((float)top_val4.s0, 1.0f);
    zerostep *= MAX_VAL_UINT;
    top_val4.s0 += zerostep;
    /* if(top_val4.s0 && top_val4.s0 < min_label) { */
    /*   min_label = top_val4.s0; */
    /* } */
    uint4 bottom_val4 = read_imageui(input_img,sampl,bottom_c);
    zerostep = (int) step((float)bottom_val4.s0, 1.0f);
    zerostep *= MAX_VAL_UINT;
    bottom_val4.s0 += zerostep;
    /* if(bottom_val4.s0 && bottom_val4.s0 < min_label) { */
    /*   min_label = bottom_val4.s0; */
    /* } */
    uint4 topleft_val4 = read_imageui(input_img,sampl,topleft_c);
    zerostep = (int) step((float)topleft_val4.s0, 1.0f);
    zerostep *= MAX_VAL_UINT;
    topleft_val4.s0 += zerostep;
    /* /\* if(topleft_val4.s0 && topleft_val4.s0 < min_label) { *\/ */
    /* /\*   min_label = topleft_val4.s0; *\/ */
    /* /\* } *\/ */
    uint4 topright_val4 = read_imageui(input_img,sampl,topright_c);
    zerostep = (int) step((float)topright_val4.s0, 1.0f);
    zerostep *= MAX_VAL_UINT;
    topright_val4.s0 += zerostep;
    /* /\* if(topright_val4.s0 && topright_val4.s0 < min_label) { *\/ */
    /* /\*   min_label = topright_val4.s0; *\/ */
    /* /\* } *\/ */
    uint4 bottomleft_val4 = read_imageui(input_img,sampl,bottomleft_c);
    zerostep = (int) step((float)bottomleft_val4.s0, 1.0f);
    zerostep *= MAX_VAL_UINT;
    bottomleft_val4.s0 += zerostep;
    /* /\* if(bottomleft_val4.s0 && bottomleft_val4.s0 < min_label) { *\/ */
    /* /\*   min_label = bottomleft_val4.s0; *\/ */
    /* /\* } *\/ */
    uint4 bottomright_val4 = read_imageui(input_img,sampl,bottomright_c);
    zerostep = (int) step((float)bottomright_val4.s0, 1.0f);
    zerostep *= MAX_VAL_UINT;
    bottomright_val4.s0 += zerostep;

    uint min_top = min(min(top_val4.s0,topright_val4.s0),topleft_val4.s0);
    uint min_center = min(min(left_val4.s0,right_val4.s0),(uint)MAX_VAL_UINT);
    uint min_bottom = min(min(bottom_val4.s0,bottomleft_val4.s0),bottomright_val4.s0);

    uint min_all = min(min(min_top,min_bottom),min_center);

    /* uint min_1 = min(min(top_val4.s0, bottom_val4.s0),(uint)MAX_VAL_UINT); */
    /* uint min_all = min(min(min_1,left_val4.s0),right_val4.s0); */
    /* if(bottomright_val.s0 && bottomright_val.s0 < min_label) { */
    /*   min_label = bottomright_val.s0; */
    /* } */

    /* If min_label is now lower than our pixel's label, change our label to
       the lowest. In the paper we are supposed to read the value from the lowest
       label we found (in case this label has changed to an even lower value by now).
       This is not possible with this approach, due to the read_only or write_only
       limitation of Image2D. Thus we're probably have to run some more iterations than
       we normally would. This could be solved by using buffers, but then we have to
       check bounds or use a padded image. Also there is no caching, so we would have
       to read coalesced (obviously only possible in the first steps, but might be enough).
       TODO: Test this?
    */
    if(min_all < my_val.s0) {
      img[pos] = min_all;
      barrier(CLK_GLOBAL_MEM_FENCE);
      uint4 out_val; 
      uint ll = img[my_val.s0];
      uint min_final = min(min_all,ll);
      img[pos] = min_final;
      out_val.s0 = min_final;
      //      my_val.s0 = min_label;
      write_imageui(output_img, my_c, out_val);
      isNotDone[0] = 1; // Since we changed a value, we need another iteration.
      // Since we are using images we have to check this value on the host,
      // reset it and then run the kernel again if it was 1.
      // This might make a big overhead? Need to take this into consideration also
      // when deciding whether or not to use images.
    }
  }
  
} 

/* Input here is output from CCL_init_labels */
__kernel void CCL_scanning_local(__read_only image2d_t input_img, __write_only image2d_t output_img,
            global uint* isNotDone, global uint* img,local uint* l_img)
{
  const sampler_t sampl = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  int x = get_global_id(0);
  int y = get_global_id(1);
  int width = get_image_width(input_img);
    
  //int pos = (y*width)+x;
  int lx = get_local_id(0);
  lx += 1;
  int ly = get_local_id(1);
  ly += 1;

  /* Step 1 - Read pixel above this work-item into local mem */
  uint4 above_val = read_imageui(input_img,sampl,(int2)(x,y-1));
  l_img[(BLOCK_WIDTH_PAD * (ly-1)) +lx] = above_val.s0;

  /* Step 2: Last line + below last line is missing, read those */
  if(ly == BLOCK_HEIGHT) {
    /* Read my item */
    uint4 tmp_val = read_imageui(input_img,sampl,(int2)(x,y));
    l_img[(BLOCK_WIDTH_PAD * ly) + lx] = tmp_val.s0;
    
    /* Read item below */
    tmp_val = read_imageui(input_img,sampl,(int2)(x,y+1));
    l_img[(BLOCK_WIDTH_PAD * (ly + 1)) + lx] = tmp_val.s0;
    
    /* Leftmost pixel? Read leftmost value of below and rightmost value of below*/
    if(lx == 1) {
      tmp_val = read_imageui(input_img,sampl,(int2)(x-1,y+1));
      l_img[(BLOCK_WIDTH_PAD * (ly + 1)) + (lx - 1)] = tmp_val.s0;
      tmp_val = read_imageui(input_img,sampl,(int2)(x+BLOCK_WIDTH+1,y+1));
      l_img[(BLOCK_WIDTH_PAD * (ly + 1)) + (lx + BLOCK_WIDTH + 1)] = tmp_val.s0;
    }
  }

  /* Step 3 - Read the rest of the edges (left and right side) */

  /* Left side */
  if(lx == 1) {
    uint4 tmp_val;
    /* Read left edge */
    tmp_val = read_imageui(input_img,sampl,(int2)(x-1,y));
    l_img[(BLOCK_WIDTH_PAD * ly) + (lx - 1)] = tmp_val.s0;
    
    /* Top pixel? Read above's left edge */
    if(ly == 1) {
      tmp_val = read_imageui(input_img,sampl,(int2)(x-1,y-1));
      l_img[(BLOCK_WIDTH_PAD * (ly - 1)) + (lx - 1)] = tmp_val.s0;
    } 
  }
 
  /* Right side */
  if(lx == BLOCK_WIDTH) {
    uint4 tmp_val;
    
    /* Read right edge */
    tmp_val = read_imageui(input_img,sampl,(int2)(x+1,y));
    l_img[(BLOCK_WIDTH_PAD * ly) + (lx + 1)] = tmp_val.s0;

    /* Top pixel, read above's right edge */
    if(ly == 1) {
      tmp_val = read_imageui(input_img, sampl, (int2)(x+1,y-1));
      l_img[(BLOCK_WIDTH_PAD * (ly - 1)) + (lx + 1)] = tmp_val.s0;
    }
  }

  /* Done reading to local memory. Synchronize */
  barrier(CLK_LOCAL_MEM_FENCE);

  //int top_left = (BLOCK_WIDTH_PAD * (ly-1)) + (lx - 1);
  int top = (BLOCK_WIDTH_PAD * (ly-1)) + lx;
  //int top_right = (BLOCK_WIDTH_PAD * (ly-1)) + (lx + 1);
  int left = (BLOCK_WIDTH_PAD * ly) + (lx - 1);
  int me = (BLOCK_WIDTH_PAD * ly) + lx;
  int right = (BLOCK_WIDTH_PAD * ly) + (lx + 1);
  int bottom = (BLOCK_WIDTH_PAD * (ly + 1)) + lx;
  //int bottom_left = (BLOCK_WIDTH_PAD * (ly + 1)) + (lx - 1);
  //int bottom_right = (BLOCK_WIDTH_PAD * (ly + 1)) + (lx + 1);

  ///
  
  int2 my_c = (int2)(x,y);
  //int2 left_c = (int2)(x-1, y);
  //int2 right_c = (int2)(x+1, y);
  //int2 top_c = (int2)(x, y-1);
  //int2 bottom_c = (int2)(x, y+1);
  //int2 topleft_c = (int2)(x-1,y-1);
  //int2 topright_c = (int2)(x+1,y-1);
  //int2 bottomleft_c = (int2)(x-1,y+1);
  //int2 bottomright_c = (int2)(x+1,y+1);

  /* /\* Get value for this thread's pixel *\/ */
  /* uint4 my_val = read_imageui(input_img, sampl, my_c); */

  /* /\* First, write this out to output_img *\/ */
  /* write_imageui(output_img, my_c, my_val); */
  /* img[pos] = my_val.s0; */

  if(x == 0 && y == 0) {
    isNotDone[0] = 0;
  }

  /* If pixel is not background: */
  if(l_img[me]) {
    //uint min_label = MAX_VAL_UINT; //TODO: fix this, now just defined something
    uint left_val = l_img[left];
    int zerostep = (int) step((float)left_val, 1.0f);
    zerostep *= MAX_VAL_UINT;
    left_val += zerostep; // if zero, add MAX_VAL_UINT
    
    /* if(left_val.s0) { */
    /*   min_label = left_val.s0; */
    /* } */
    uint right_val = l_img[right];
    zerostep = (int) step((float)right_val, 1.0f);
    zerostep *= MAX_VAL_UINT;
    right_val += zerostep;
    /* if(right_val && right_val < min_label) { */
    /*   min_label = right_val; */
    /* } */
    uint top_val = l_img[top];
    zerostep = (int) step((float)top_val, 1.0f);
    zerostep *= MAX_VAL_UINT;
    top_val += zerostep;
    /* if(top_val && top_val < min_label) { */
    /*   min_label = top_val; */
    /* } */
    uint bottom_val = l_img[bottom];
    zerostep = (int) step((float)bottom_val, 1.0f);
    zerostep *= MAX_VAL_UINT;
    bottom_val += zerostep;
    /* if(bottom_val && bottom_val < min_label) { */
    /*   min_label = bottom_val4.s0; */
    /* } */
    /* uint4 topleft_val4 = read_imageui(input_img,sampl,topleft_c); */
    /* zerostep = (int) step((float)topleft_val4.s0, 1.0f); */
    /* zerostep *= MAX_VAL_UINT; */
    /* topleft_val4.s0 += zerostep; */
    /* /\* if(topleft_val4.s0 && topleft_val4.s0 < min_label) { *\/ */
    /* /\*   min_label = topleft_val4.s0; *\/ */
    /* /\* } *\/ */
    /* uint4 topright_val4 = read_imageui(input_img,sampl,topright_c); */
    /* zerostep = (int) step((float)topright_val4.s0, 1.0f); */
    /* zerostep *= MAX_VAL_UINT; */
    /* topright_val4.s0 += zerostep; */
    /* /\* if(topright_val4.s0 && topright_val4.s0 < min_label) { *\/ */
    /* /\*   min_label = topright_val4.s0; *\/ */
    /* /\* } *\/ */
    /* uint4 bottomleft_val4 = read_imageui(input_img,sampl,bottomleft_c); */
    /* zerostep = (int) step((float)bottomleft_val4.s0, 1.0f); */
    /* zerostep *= MAX_VAL_UINT; */
    /* bottomleft_val4.s0 += zerostep; */
    /* /\* if(bottomleft_val4.s0 && bottomleft_val4.s0 < min_label) { *\/ */
    /* /\*   min_label = bottomleft_val4.s0; *\/ */
    /* /\* } *\/ */
    /* uint4 bottomright_val4 = read_imageui(input_img,sampl,bottomright_c); */
    /* zerostep = (int) step((float)bottomright_val4.s0, 1.0f); */
    /* zerostep *= MAX_VAL_UINT; */
    /* bottomright_val4.s0 += zerostep; */

    /* uint min_top = min(min(top_val4.s0,topright_val4.s0),topleft_val4.s0); */
    /* uint min_center = min(min(left_val4.s0,right_val4.s0),(uint)MAX_VAL_UINT); */
    /* uint min_bottom = min(min(bottom_val4.s0,bottomleft_val4.s0),bottomright_val4.s0); */

    /* uint min_all = min(min(min_top,min_bottom),min_center); */

    uint min_1 = min(min(top_val, bottom_val),(uint)MAX_VAL_UINT);
    uint min_all = min(min(min_1,left_val),right_val);
    /* if(bottomright_val.s0 && bottomright_val.s0 < min_label) { */
    /*   min_label = bottomright_val.s0; */
    /* } */

    /* If min_label is now lower than our pixel's label, change our label to
       the lowest. In the paper we are supposed to read the value from the lowest
       label we found (in case this label has changed to an even lower value by now).
       This is not possible with this approach, due to the read_only or write_only
       limitation of Image2D. Thus we're probably have to run some more iterations than
       we normally would. This could be solved by using buffers, but then we have to
       check bounds or use a padded image. Also there is no caching, so we would have
       to read coalesced (obviously only possible in the first steps, but might be enough).
       TODO: Test this?
    */
    if(min_all < l_img[me]) {
      //      img[pos] = min_all;
      //      barrier(CLK_GLOBAL_MEM_FENCE);
      uint4 out_val; 
      /* uint ll = img[l_img[me]]; */
      /* uint min_final = min(min_all,ll); */
      /* img[pos] = min_final; */
      out_val.s0 = min_all;
      //      my_val.s0 = min_label;
      write_imageui(output_img, my_c, out_val);
      isNotDone[0] = 1; // Since we changed a value, we need another iteration.
      // Since we are using images we have to check this value on the host,
      // reset it and then run the kernel again if it was 1.
      // This might make a big overhead? Need to take this into consideration also
      // when deciding whether or not to use images.
    } else {
      uint4 my_val;
      my_val.s0 = l_img[me];
      write_imageui(output_img, my_c, my_val);
      //    img[pos] = my_val.s0;
    }
  }
  
} 

__kernel void CCL_analysis(__read_only image2d_t input_image, global uint* output_image) {
  const sampler_t sampl = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  
  int x = get_global_id(0);
  int y = get_global_id(1);
  int width = get_image_width(input_image);
  int pos = (y*width)+x;

  /* Copy labels to output_image */
  int2 coords = (int2)(x,y);
  uint4 label4 = read_imageui(input_image, sampl, coords);

  /* Sync? */
  output_image[pos] = label4.s0;
  uint label = label4.s0;

  barrier(CLK_GLOBAL_MEM_FENCE);

  if(label) {
    uint r = output_image[label];
    while(r != label) {
      label = output_image[r];
      r = output_image[label];
    }
    output_image[pos] = label;
  }
  //barrier(CLK_GLOBAL_MEM_FENCE);
  
  // Testing, set all output pixels to 255 (white).
  /* if(output_image[pos] > 0) { */
  /* /\* if(label > 0) { *\/ */
  /*   output_image[pos] = 255; */
  /* } */
}

__kernel void CCL_calculate_size() {
}

/* Begin local mem close-open
   Closing and opening performed in one kernel.
   global memory as input read into local memory.
   Global memory is also used between the different
   steps to synchronize.
*/
__kernel void close_and_open_image(__global uchar* input_img, global uchar* output_img, local uchar* l_img) {
  uint width = get_global_size(0);
  uint height = get_global_size(1);
  uint block_width = get_local_size(0);
  uint block_height = get_local_size(1);
  /* 1 pixel padding around the image for borders */
  uint block_width_pad = block_width + 2;
  //uint block_height_pad = block_height + 2;
  uint x = get_global_id(0);
  uint y = get_global_id(1);
  uint global_id = y*width+x;

  /* Local x and y */
  uint lx = get_local_id(0);
  uint ly = get_local_id(1);
  lx = lx + 1; // Compensate for padding
  ly = ly + 1; // of local mem
  uint local_id = (block_width_pad * ly) + lx;

  /* Read my value to local mem */
  l_img[local_id] = input_img[global_id];

  /* Read border regions (padding) to local memory */
  read_border_regions(input_img, l_img,
		      width, height, x, y, 
		      lx, ly, block_width,
		      block_height, 0);  

  /* Done reading to local memory. Synchronize */
  barrier(CLK_LOCAL_MEM_FENCE);

  /* 1st dilate */
  uchar max_all = find_max_value(l_img, block_width_pad, lx, ly);
  barrier(CLK_LOCAL_MEM_FENCE);
  l_img[local_id] = max_all; // store value in local mem

  /* Sync to global mem */
  input_img[global_id] = max_all;
  barrier(CLK_GLOBAL_MEM_FENCE);

  /* Read border regions again, now with 255 as default val
     for regions outside the image. Read from output_img. */
  read_border_regions(input_img, l_img,
		      width, height, x, y, 
		      lx, ly, block_width,
		      block_height, 255);

  /* Done reading to local memory. Synchronize */
  barrier(CLK_LOCAL_MEM_FENCE);
    
  /* 1st erode */
  uchar min_all = find_min_value(l_img, block_width_pad, lx, ly);
  barrier(CLK_LOCAL_MEM_FENCE);
  l_img[local_id] = min_all; // store in local mem

  /* Sync to global mem */
  input_img[global_id] = min_all;
  barrier(CLK_GLOBAL_MEM_FENCE);

  /* Read border regions again.*/
  read_border_regions(input_img, l_img,
  		      width, height, x, y,
  		      lx, ly, block_width,
  		      block_height, 255);

  /* Done reading to local memory. Synchronize */
  barrier(CLK_LOCAL_MEM_FENCE);
  
  /* 2nd erode */
  min_all = find_min_value(l_img, block_width_pad, lx, ly);
  barrier(CLK_LOCAL_MEM_FENCE);
  l_img[local_id] = min_all; // store in local mem

  /* Sync to global mem */
  input_img[global_id] = min_all;
  barrier(CLK_GLOBAL_MEM_FENCE);

  /* Read border regions again (now with 0 as default_val,
     as we will dilate this time */
  read_border_regions(input_img, l_img,
  		      width, height, x, y,
  		      lx, ly, block_width,
  		      block_height, 0);

  /* Done reading to local memory. Synchronize */
  barrier(CLK_LOCAL_MEM_FENCE);

  /* 2nd dilate */
  max_all = find_max_value(l_img, block_width_pad, lx, ly);
  /* Last step - no need to store in local mem */

  /* Finally, write out result to global mem */
  output_img[global_id] = max_all;
}

/* END localmem-close-open */

__kernel void dilate_image_local(__read_only  image2d_t input_img, 
            __write_only image2d_t output_img, local uint *l_img) {
 
  /* Sampler for image */
  const sampler_t sampl = CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP |
    CLK_FILTER_NEAREST;

  int block_width = get_local_size(0);
  int block_height = get_local_size(1);
  int block_width_pad = block_width + 2;
  //int block_height_pad = block_height + 2;
  int x = get_global_id(0);
  int y = get_global_id(1);
  int lx = get_local_id(0);
  lx += 1;
  int ly = get_local_id(1);
  ly += 1;

  /* Step 1 - Read pixel above this work-item into local mem */
  uint4 above_val = read_imageui(input_img,sampl,(int2)(x,y-1));
  l_img[(block_width_pad * (ly-1)) +lx] = above_val.s0;

  /* Step 2: Last line + below last line (border) is missing, read those */
  if(ly == block_height) {
    /* Read my item (last line) */
    uint4 tmp_val = read_imageui(input_img,sampl,(int2)(x,y));
    l_img[(block_width_pad * ly) + lx] = tmp_val.s0;
    
    /* Read item below */
    tmp_val = read_imageui(input_img,sampl,(int2)(x,y+1));
    l_img[(block_width_pad * (ly + 1)) + lx] = tmp_val.s0;
    
    /* Leftmost pixel? Read leftmost value of below */
    if(lx == 1) {
      tmp_val = read_imageui(input_img,sampl,(int2)(x-1,y+1));
      l_img[(block_width_pad * (ly + 1)) + (lx - 1)] = tmp_val.s0;
      /* tmp_val = read_imageui(input_img,sampl,(int2)(x+block_width+1,y+1)); */
      /* l_img[(block_width_pad * (ly + 1)) + (lx + block_width + 1)] = tmp_val.s0; */
    }
    /* Rightmost pixel? Opposite of above (rightmost value of below) */
    if(lx == block_width) {
      tmp_val = read_imageui(input_img,sampl,(int2)(x+1,y+1));
      l_img[(block_width_pad * (ly + 1)) + (lx + 1)] = tmp_val.s0;
    }
  }

  /* Step 3 - Read the rest of the edges (left and right side) */

  /* Left side */
  if(lx == 1) {
    uint4 tmp_val;
    /* Read left edge */
    tmp_val = read_imageui(input_img,sampl,(int2)(x-1,y));
    l_img[(block_width_pad * ly) + (lx - 1)] = tmp_val.s0;
    
    /* Top pixel? Read above's left edge */
    if(ly == 1) {
      tmp_val = read_imageui(input_img,sampl,(int2)(x-1,y-1));
      l_img[(block_width_pad * (ly - 1)) + (lx - 1)] = tmp_val.s0;
    } 
  }
 
  /* Right side */
  if(lx == block_width) {
    uint4 tmp_val;
    
    /* Read right edge */
    tmp_val = read_imageui(input_img,sampl,(int2)(x+1,y));
    l_img[(block_width_pad * ly) + (lx + 1)] = tmp_val.s0;

    /* Top pixel, read above's right edge */
    if(ly == 1) {
      tmp_val = read_imageui(input_img, sampl, (int2)(x+1,y-1));
      l_img[(block_width_pad * (ly - 1)) + (lx + 1)] = tmp_val.s0;
    }
  }  

  /* Done reading to local memory. Synchronize */
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Find max value */

  uint max_top = 0;
  uint max_center = 0;
  uint max_bottom = 0;

  /* Variables for more readable img-positions */
  int top_left = (block_width_pad * (ly-1)) + (lx - 1);
  int top = (block_width_pad * (ly-1)) + lx;
  int top_right = (block_width_pad * (ly-1)) + (lx + 1);
  int left = (block_width_pad * ly) + (lx - 1);
  int me = (block_width_pad * ly) + lx;
  int right = (block_width_pad * ly) + (lx + 1);
  int bottom = (block_width_pad * (ly + 1)) + lx;
  int bottom_left = (block_width_pad * (ly + 1)) + (lx - 1);
  int bottom_right = (block_width_pad * (ly + 1)) + (lx + 1);

  max_top = max(max(l_img[top_left],l_img[top]),l_img[top_right]);
  max_center = max(max(l_img[left],l_img[me]),l_img[right]);
  max_bottom = max(max(l_img[bottom_left],l_img[bottom]),l_img[bottom_right]);
  
  /* TODO: test */
  uint4 max_all;
  max_all.s0 = max(max(max_top,max_center),max_bottom);

  /* End test */

  /* Coordinates for pixels we need to check.
     Since we are using an Image2D, we don't bother
     checking boundaries. Implementation isn't thought to
     be fast atm. but the Image2D-caching should hopefully
     help a bit. */
  int2 my_c = (int2)(x, y);
  /* int2 left_c = (int2)(x-1, y); */
  /* int2 right_c = (int2)(x+1, y); */
  /* int2 top_c = (int2)(x, y-1); */
  /* int2 bottom_c = (int2)(x, y+1); */
  /* int2 topleft_c = (int2)(x-1,y-1); */
  /* int2 topright_c = (int2)(x+1,y-1); */
  /* int2 bottomleft_c = (int2)(x-1,y+1); */
  /* int2 bottomright_c = (int2)(x+1,y+1); */

  /* uint4 my_val = read_imageui(input_img,sampl,my_c); */
  /* uint4 left_val = read_imageui(input_img,sampl,left_c); */
  /* uint4 right_val = read_imageui(input_img,sampl,right_c); */
  /* uint4 top_val = read_imageui(input_img,sampl,top_c); */
  /* uint4 bottom_val = read_imageui(input_img,sampl,bottom_c); */
  /* uint4 topleft_val = read_imageui(input_img,sampl,topleft_c); */
  /* uint4 topright_val = read_imageui(input_img,sampl,topright_c); */
  /* uint4 bottomleft_val = read_imageui(input_img,sampl,bottomleft_c); */
  /* uint4 bottomright_val = read_imageui(input_img,sampl,bottomright_c); */

  /* uint max_top = 0; */
  /* uint max_center = 0; */
  /* uint max_bottom = 0; */
  
  /* max_top = max(max(topleft_val.s0,top_val.s0),topright_val.s0); */
  /* max_center = max(max(left_val.s0,my_val.s0),right_val.s0); */
  /* max_bottom = max(max(bottomleft_val.s0,bottom_val.s0),bottomright_val.s0); */
  
  /* uint4 max_all; */
  /* max_all.s0 = max(max(max_top,max_center),max_bottom); */
    
  //  barrier(CLK_LOCAL_MEM_FENCE);

  /* Write out */
  write_imageui(output_img,my_c,max_all);
}

__kernel void erode_image_local(__read_only  image2d_t input_img, 
			   __write_only image2d_t output_img, local uint *l_img) {
 
  /* Sampler for image */
  const sampler_t sampl = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  int block_width = get_local_size(0);
  int block_height = get_local_size(1);
  int block_width_pad = block_width + 2;
    
  //int block_height_pad = block_height + 2;
  int x = get_global_id(0);
  int y = get_global_id(1);
  int lx = get_local_id(0);
  lx += 1;
  int ly = get_local_id(1);
  ly += 1;

  /* Step 1 - Read pixel above this work-item into local mem */
  uint4 above_val = read_imageui(input_img,sampl,(int2)(x,y-1));
  l_img[(block_width_pad * (ly-1)) +lx] = above_val.s0;

  /* Step 2: Last line + below last line is missing, read those */
  if(ly == block_height) {
    /* Read my item */
    uint4 tmp_val = read_imageui(input_img,sampl,(int2)(x,y));
    l_img[(block_width_pad * ly) + lx] = tmp_val.s0;
    
    /* Read item below */
    tmp_val = read_imageui(input_img,sampl,(int2)(x,y+1));
    l_img[(block_width_pad * (ly + 1)) + lx] = tmp_val.s0;

    /* Leftmost pixel? Read leftmost value of line below */
    if(lx == 1) {
      tmp_val = read_imageui(input_img,sampl,(int2)(x-1,y+1));
      l_img[(block_width_pad * (ly + 1)) + (lx - 1)] = tmp_val.s0;
    }
    /* Rightmost pixel? Read rightmost value of line below.*/
    if(lx == block_width) {
      tmp_val = read_imageui(input_img,sampl,(int2)(x+1,y+1));
      l_img[(block_width_pad * (ly + 1)) + (lx + 1)] = tmp_val.s0;
    }
  }

  /* Step 3 - Read the rest of the edges (left and right side) */

  /* Left side */
  if(lx == 1) {
    uint4 tmp_val;
    /* Read left edge */
    tmp_val = read_imageui(input_img,sampl,(int2)(x-1,y));
    l_img[(block_width_pad * ly) + (lx - 1)] = tmp_val.s0;
    
    /* Top pixel? Read above's left edge */
    if(ly == 1) {
      tmp_val = read_imageui(input_img,sampl,(int2)(x-1,y-1));
      l_img[(block_width_pad * (ly - 1)) + (lx - 1)] = tmp_val.s0;
    } 
  }
 
  /* Right side */
  if(lx == block_width) {
    uint4 tmp_val;
    
    /* Read right edge */
    tmp_val = read_imageui(input_img,sampl,(int2)(x+1,y));
    l_img[(block_width_pad * ly) + (lx + 1)] = tmp_val.s0;

    /* Top pixel, read above's right edge */
    if(ly == 1) {
      tmp_val = read_imageui(input_img, sampl, (int2)(x+1,y-1));
      l_img[(block_width_pad * (ly - 1)) + (lx + 1)] = tmp_val.s0;
    }
  }

  /* Done reading to local memory. Synchronize */
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Find min value */

  uint min_top = 0;
  uint min_center = 0;
  uint min_bottom = 0;

  /* Variables for more readable img-positions */
  int top_left = (block_width_pad * (ly-1)) + (lx - 1);
  int top = (block_width_pad * (ly-1)) + lx;
  int top_right = (block_width_pad * (ly-1)) + (lx + 1);
  int left = (block_width_pad * ly) + (lx - 1);
  int me = (block_width_pad * ly) + lx;
  int right = (block_width_pad * ly) + (lx + 1);
  int bottom = (block_width_pad * (ly + 1)) + lx;
  int bottom_left = (block_width_pad * (ly + 1)) + (lx - 1);
  int bottom_right = (block_width_pad * (ly + 1)) + (lx + 1);

  min_top = min(min(l_img[top_left],l_img[top]),l_img[top_right]);
  min_center = min(min(l_img[left],l_img[me]),l_img[right]);
  min_bottom = min(min(l_img[bottom_left],l_img[bottom]),l_img[bottom_right]);
  
  /* TODO: test */
  uint4 min_all;
  min_all.s0 = min(min(min_top,min_center),min_bottom);

  /* End test */

  int2 my_c = (int2)(x, y);
  /* Write out */
  write_imageui(output_img,my_c,min_all);
}


/* CCL-Algorithm from GPU Computing Gems book
   Based on reading the cuda code from:
   http://www.gpucomputing.net/?q=node/1312
   and reimplementing for OpenCL.
   Not done yet - not sure if the kernels are OK.
   Might also contain some debug-code.
   
   input = segmented input data (whole image)
   segs = local(shared) storage for segmented input data (0 or 1)
   labels = local(shared) storage for labels
   output = assigned label for each pixel (whole image)
*/
__kernel void CCL_kernel_1(global uchar* input, global int* output, local uchar* segs, local int* labels,
            local uint* changed,int width)
{
  int block_width = get_local_size(0);
  int block_height = get_local_size(1);
  int block_width_pad = block_width + 2;
  int block_height_pad = block_height + 2;

  int lx = get_local_id(0);
  lx += 1; // Remove if we're not using padding. Same below.
  // Also, change block_width_pad to block_width in that case.
  int ly = get_local_id(1);
  ly += 1;

  int x = get_global_id(0);
  int y = get_global_id(1);

  int localIndex = (block_width_pad * ly) + lx;
  int globalIndex = (y*width)+x;

  int newLabel = localIndex;
  int oldLabel = 0;

  /* Get data into local mem */
  segs[localIndex] = input[globalIndex];
  barrier(CLK_LOCAL_MEM_FENCE);
  
  /* Zero out border pixels around the tile
     we're working at. */

  /* Left side */
  if(lx == 1) {
    segs[localIndex-1] = 0;
    labels[localIndex-1] = 0;
  }

  /* Top */
  if(ly == 1) {
    segs[((ly-1)*block_width_pad) + lx] = 0;
    labels[((ly-1)*block_width_pad) + lx] = 0;
  }

  /* Bottom */
  if(ly == (block_height_pad-1)) {
    segs[((ly+1)*block_width_pad)+lx] = 0;
    labels[((ly+1)*block_width_pad)+lx] = 0;
  }
  
  /* Right side */
  if(lx == (block_width_pad-1)) {
    segs[localIndex+1] = 0;
    labels[localIndex+1] = 0;
  }

  /* Corners */
  if(lx == 1 && ly == 1) {
    segs[((ly-1)*block_width_pad)+(lx-1)] = 0;
    labels[((ly-1)*block_width_pad)+(lx-1)] = 0;
  }
  if(lx == (block_width_pad-1) && ly == 1) {
    segs[((ly-1)*block_width_pad)+(lx+1)] = 0;
    labels[((ly-1)*block_width_pad)+(lx+1)] = 0;
  }
  if(lx == 1 && ly == (block_height_pad-1)) {
    segs[((ly+1)*block_width_pad)+(lx-1)] = 0;
    labels[((ly+1)*block_width_pad)+(lx-1)] = 0;
  }
  if(lx == (block_width_pad-1) && ly == (block_height_pad-1)) {
    segs[((ly+1)*block_width_pad)+(lx+1)] = 0;
    labels[((ly+1)*block_width_pad)+(lx+1)] = 0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  //int label = localIndex;
  //  uint oldLabel = 0;

  uchar seg = segs[localIndex];
  
  while(1) {
    // Pass 1 - update the label. First pass, label is 
    // equal to the local addr. of the element
    labels[localIndex] = newLabel;
    
    /* Reset check flag */
    if(lx == 1 && ly == 1) {
      changed[0] = 0;
    }
    oldLabel = newLabel;
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Only compare if elem is not background */
    if(seg != 0) {
      /* Compare the element's label with its neighbors
	 and find minimal label from neighbors (Moore-neighborhood) */

      /* Variables for readable neighbour positions.
	 Can be dropped if needed.*/
      int top_left = (block_width_pad * (ly-1)) + (lx - 1);
      int top = (block_width_pad * (ly-1)) + lx;
      int top_right = (block_width_pad * (ly-1)) + (lx + 1);
      int left = (block_width_pad * ly) + (lx - 1);
      //int me = (block_width_pad * ly) + lx;
      int right = (block_width_pad * ly) + (lx + 1);
      int bottom = (block_width_pad * (ly + 1)) + lx;
      int bottom_left = (block_width_pad * (ly + 1)) + (lx - 1);
      int bottom_right = (block_width_pad * (ly + 1)) + (lx + 1);

      if(segs[localIndex] == segs[top_left])
        newLabel = min(newLabel,labels[top_left]);
      if(segs[localIndex] == segs[top])
        newLabel = min(newLabel,labels[top]);
      if(segs[localIndex] == segs[top_right])
        newLabel = min(newLabel,labels[top]);
      if(segs[localIndex] == segs[left])
        newLabel = min(newLabel,labels[left]);
      if(segs[localIndex] == segs[right])
        newLabel = min(newLabel,labels[right]);
      if(segs[localIndex] == segs[bottom])
        newLabel = min(newLabel,labels[bottom]);
      if(segs[localIndex] == segs[bottom_left])
        newLabel = min(newLabel,labels[bottom_left]);
      if(segs[localIndex] == segs[bottom_right])
        newLabel = min(newLabel,labels[bottom_right]);
    }
      
    barrier(CLK_LOCAL_MEM_FENCE);
    
    /* New label smaller than hte old ones? Merge equivalence trees */
    if(oldLabel > newLabel) {
      atom_min(labels+oldLabel,newLabel); //TODO: check this in example code.
      changed[0] = 1;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Local solution found? Exit loop
    if(changed[0] == 0) {
      break;
    }
    
    /* Pass 2 - flatten the equivalence tree */
    newLabel = findRoot(labels, newLabel);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  /* segmented data is 0, meaning background. set label to -1 */
  if(seg == 0) {
    newLabel = -1;
  }
  else {
    ly -=1;
    lx -=1;
    /* Transfer label into global coords and
       store result into global device memory */
    y = newLabel / block_width_pad;
    x = newLabel - y*block_width_pad;
    x = lx*block_width + x-1;
    y = ly*block_width + y-1;
    newLabel = x+y*width;
  }
  output[globalIndex] = newLabel;
}


__kernel void CCL_kernel_2_testing(global int* labelsInOut, __read_only image2d_t seg_in, local uint* changed,
            int width, int tileDim)
{
  //int tileX = get_local_id(0)+get_group_id(0) * get_local_size(0);
  if(get_local_id(0) == 4 && get_local_id(1) == 4 && get_local_id(2) == 15) {
    labelsInOut[0] = get_global_id(0);
  }
}

/* CCL-Algorithm from GPU Computing Gems book - Kernel 2
*  Merge equivalence trees on border kernels 
labelsInOut = labels from previous step and previous iterations of this kernel
seg_in = segmented input data (0 or 1) in image2d_t
*/
__kernel void CCL_kernel_2(global int* labelsInOut, read_only image2d_t seg_in, local uint* changed, int width,
            int tileDim)
{
  /* Sampler for image */
  const sampler_t sampl = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  // int2 coord = (x,y);
  //  uint4 my_val = read_imageui(seg_in,sampl,coord);
  
  int lx = get_local_id(0);
  int ly = get_local_id(1);
  int lz = get_local_id(2);

  /* Directly mapped to image space */
  int tileX = lx + get_group_id(0) * get_local_size(0);
  int tileY = ly + get_group_id(1) * get_local_size(1);

  /* Number of iterations each thread has to perform for one border of the tile */
  int threadIterations = tileDim / get_local_size(2);// z dimension

  // dimensions of the tile in the next merging-step
  int nextTileDim = tileDim * get_local_size(0);
  
  uchar seg;
  int offset;

  int countTest = 0;

  while(1) {
    countTest++;
    // Reset check variable
    if(lx == 0 && ly == 0 && lz == 0) {
      changed[0] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    /* 1: Process horizontal borders between merged tiles */
    if(ly < get_local_size(1)-1) {
      /* Horizontal border is the last row of the tile */
      uint y = (tileY+1)*tileDim-1;
      
      /* Offset of the element from the leftmost boundary of the tile */
      offset = lx*tileDim + lz;
      uint x = tileX*tileDim + lz;

      for(int i=0;i<threadIterations;++i) {
	
      	/* Load segment data for the element */
      	uint4 seg4 = read_imageui(seg_in,sampl,(int2)(x,y));
      	seg = (uchar) seg4.s0;
      	
      	if(seg != 0) {
      	  /* Address of elem in global space */
      	  int idx = x+y*width;

      	  // perform union operation on neighboring elems from other tiles that are to
      	  // be merged with the processed tile

      	  if(offset>0) {
      	    uint4 seg4_tmp = read_imageui(seg_in,sampl,(int2)((x-1),(y+1)));
      	    unionF(labelsInOut, seg, (uchar)seg4_tmp.s0, idx, idx-1+width, changed);
      	  }

      	  uint4 seg4_tmp = read_imageui(seg_in,sampl,(int2)(x,(y+1)));
      	  unionF(labelsInOut, seg, (uchar)seg4_tmp.s0, idx, idx+width, changed);

      	  if(offset<nextTileDim-1) {
      	    uint4 seg4_tmp2 = read_imageui(seg_in,sampl,(int2)((x+1),(y+1)));
      	    unionF(labelsInOut, seg, (uchar)seg4_tmp2.s0, idx, idx+1+width, changed);
      	  }
      	}
  	
    	  // set the processed element to the next in line on the same boundary ...
      	x += get_local_size(2);
      	offset += get_local_size(2);
      }
    }

    
    /* 2: Process vertical borders between merged tiles */
    if(lx < get_local_size(0)-1) {
      /* Vertical border is the rightmost column of elements in the tile */
      uint x = (tileX+1)*tileDim-1;

      // offset of the element from the topmost boundary of the tile
      offset = ly*tileDim+lz;
      
      uint y = tileY*tileDim+lz;

      for(int i=0;i<threadIterations;++i) {
      	uint4 seg4 = read_imageui(seg_in,sampl,(int2)(x,y));
      	seg = (uchar)seg4.s0;
	
      	if(seg != 0) {
      	  int idx = x+y*width;
      	  
      	  // perform union operation on neighboring elems from other tiles
      	  // that are to be merged with the processed tile
      	  if(offset > 0) {
      	    uint4 seg4_tmp = read_imageui(seg_in,sampl,(int2)((x+1),(y-1)));
      	    unionF(labelsInOut, seg, (uchar)seg4_tmp.s0, idx, idx+1-width, changed);
      	  }
      	  uint4 seg4_tmp = read_imageui(seg_in,sampl,(int2)((x+1),y));
      	  unionF(labelsInOut, seg, (uchar)seg4_tmp.s0, idx, idx+1, changed);
      	  
      	  if(offset < nextTileDim-1) {
      	    uint4 seg4_tmp2 = read_imageui(seg_in,sampl,(int2)((x+1),(y+1)));
      	    unionF(labelsInOut, seg, (uchar)seg4_tmp2.s0, idx, idx+1+width, changed);
      	  } 
      	}
      	y += get_local_size(2);
      	offset += get_local_size(2);
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    /* If no equivalence tree was updated, then all equivalence trees of the merged
       tiles are already merged */

    if(changed[0] == 0) {
      break;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

// TODO: check kernel 2 for errors and shite. check TODO's.
// also need to check how to execute it.


/* CCL-Algorithm from GPU Computing Gems book - Kernel 3
   Flatten equivalence trees after merging tiles.
   Called in between kernel 2 calls.
*/
__kernel void CCL_kernel_3(global int* labelsInOut, __read_only image2d_t seg_in, local uint* changed, int width, int log2width, int tileDim, int blocksPerTile)
{

  // multiple work groups can be used to update border of a single tile
  int tileX = get_group_id(0) / blocksPerTile;
  int tileOffset = get_local_size(0)*(get_group_id(0) & (blocksPerTile-1));
  int tileY = get_local_id(1) + (get_group_id(1)*get_local_size(1));
  int maxTileY = get_global_size(1)*get_local_size(1)-1;

  /* a single thread is used to update both the horizontal and
     the vertical boundary on both sides of the two merged tiles */

  // first process horizontal borders
  if(tileY < maxTileY) {
    uint y = (tileY+1)*tileDim-1;
    uint x = tileX*tileDim+get_local_id(0)+tileOffset;
    
    flattenEquivalenceTreesInternal(x,y,labelsInOut,labelsInOut,width,log2width);
    flattenEquivalenceTreesInternal(x,y+1,labelsInOut,labelsInOut,width,log2width);
  }
  
  // process vertical borders
  if(tileX < get_global_id(0)-1) {
    uint x = (tileX+1)*tileDim-1;
    uint y = tileY*tileDim+get_local_id(0)+tileOffset;
    flattenEquivalenceTreesInternal(x,y,labelsInOut,labelsInOut,width,log2width);
    flattenEquivalenceTreesInternal(x+1,y,labelsInOut,labelsInOut,width,log2width);
  }
}
/* End kernel 3 */

/* CCL-Algorithm from GPU Computing Gems book - Kernel 4
   Complete flattening of equivalence trees */
__kernel void CCL_kernel_4(global int* labelsInOut, 
			   /* global int* labelsIn, */
			   int width, int log2width) {
  // flatten equivalence trees on all elements
  uint x = (get_group_id(0)*get_local_size(0))+get_local_id(0);
  uint y = (get_group_id(1)*get_local_size(1))+get_local_id(1);
  
  flattenEquivalenceTreesInternal(x,y,labelsInOut,labelsInOut,width,log2width);
}

/* End kernel 4 */

/* CCL-Algorithm from GPU Computing Gems book - Reindex kernels.
   Kernels used for reindexing the labels */
__kernel void prepareLabelsForScan(global int* outData, global int* labelsIn, int width)
{
  uint x = (get_group_id(0)*get_local_size(0))+get_local_id(0);
  uint y = (get_group_id(1)*get_local_size(1))+get_local_id(1);
  
  int index = x+(y*width);
  int label = labelsIn[index];
  int ret = 0;
  if(label == index)
    ret = 1;
  outData[index] = ret;
}

__kernel void reindexLabels(global int* labelsOut, global int* labelsIn, global int* scanIn, int width,
            int numComponents)
{
  uint x = (get_group_id(0)*get_local_size(0))+get_local_id(0);
  uint y = (get_group_id(1)*get_local_size(1))+get_local_id(1);
  
  int index = x+(y*width);
  int label = labelsIn[index];
  int ret = -1;
  if(label != -1) {
    int scanId = scanIn[label]; // label uses same index as scanned data
    ret = numComponents - scanId;
  }
  labelsOut[index] = ret;
}

/* End CCL reindex kernels */


