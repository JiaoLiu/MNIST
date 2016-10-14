//
//  MLCnn.h
//  MNIST
//
//  Created by Jiao Liu on 9/28/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>

@interface MLCnn : NSObject
{
    @private
    NSArray *_filters;
    int _connectSize;
    int _numOfFilter;
    int _dimRow;
    int _dimCol;
    int _outRow;
    int _outCol;
    double _keepProb;
    double **_weight;
    double **_bias;
    double **_filteredImage;
    double **_reluFlag;
    double *_dropoutMask;
}

+ (double *)weight_init:(int)size;
+ (double *)bias_init:(int)size;
- (id)initWithFilters:(NSArray *)filters fullConnectSize:(int)size row:(int)dimRow col:(int)dimCol keepRate:(double)rate;
- (double *)filterImage:(double *)image state:(BOOL)isTraining;
- (void)backPropagation:(double *)loss;

@end
