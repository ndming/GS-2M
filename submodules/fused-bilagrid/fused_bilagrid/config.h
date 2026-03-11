#ifndef _CONFIG_H_INC
#define _CONFIG_H_INC

// RGB to gray constants
#define kC2G_r 0.299f
#define kC2G_g 0.587f
#define kC2G_b 0.114f

#if 0
#include "stdio.h"
#define CHECK_IDX(idx, max_size) \
    if ((idx) < 0 || (idx) >= (max_size)) { \
        printf("out of bound: %s @ %d, %d/%d\n", __FILE__, __LINE__, idx, max_size); \
    }
#endif

#endif  // _CONFIG_H_INC
