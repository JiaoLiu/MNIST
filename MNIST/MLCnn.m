//
//  MLCnn.m
//  MNIST
//
//  Created by Jiao Liu on 9/28/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import "MLCnn.h"

@implementation MLCnn

+ (double)truncated_normal:(double)mean dev:(double)stddev
{
    double outP = 0.0;
    do {
        static int hasSpare = 0;
        static double spare;
        if (hasSpare) {
            hasSpare = 0;
            outP = mean + stddev * spare;
            continue;
        }
        
        hasSpare = 1;
        static double u,v,s;
        do {
            u = (rand() / ((double) RAND_MAX)) * 2.0 - 1.0;
            v = (rand() / ((double) RAND_MAX)) * 2.0 - 1.0;
            s = u * u + v * v;
        } while ((s >= 1.0) || (s == 0.0));
        s = sqrt(-2.0 * log(s) / s);
        spare = v * s;
        outP = mean + stddev * u * s;
    } while (fabsl(outP) > 2*stddev);
    return outP;
}

- (double)relu:(double)x
{
    return MAX(0, x);
}

# pragma mark - CNN Main



@end
