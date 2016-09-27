//
//  MLSoftMax.h
//  MNIST
//
//  Created by Jiao Liu on 9/26/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>

@interface MLSoftMax : NSObject
{
    @private
    int _iterNum;
    int _kType;
    int _dim;
    int _randSize;
    double **_randomX;
    int *_randomY;
    double *_theta;
    double *_bias;
}

@property (nonatomic, assign) double **image;
@property (nonatomic, assign) int *label;
@property (nonatomic, assign) int trainNum;
@property (nonatomic, assign) double descentRate;

- (id)initWithLoopNum:(int)loopNum dim:(int)dim type:(int)type size:(int)size;
- (void)train;
- (int)predict:(double *)image;

@end
