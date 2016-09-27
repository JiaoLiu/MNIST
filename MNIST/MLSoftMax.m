//
//  MLSoftMax.m
//  MNIST
//
//  Created by Jiao Liu on 9/26/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import "MLSoftMax.h"

@implementation MLSoftMax

- (id)initWithLoopNum:(int)loopNum dim:(int)dim type:(int)type size:(int)size descentRate:(double)rate
{
    self = [super init];
    if (self) {
        _iterNum = loopNum == 0 ? 500 : loopNum;
        _dim = dim;
        _kType = type;
        _randSize = size == 0 ? 100 : size;
        _bias = malloc(sizeof(double) * type);
        _theta = malloc(sizeof(double) * type * dim);
        for (int i = 0; i < type; i++) {
            _bias[i] = 0;
            for (int j = 0; j < dim; j++) {
                _theta[i * dim +j] = 0.0f;
            }
        }
        
        _descentRate = rate == 0 ? 0.01 : rate;
    }
    return  self;
}

- (void)dealloc
{
    if (_bias != NULL) {
        free(_bias);
        _bias = NULL;
    }
    
    if (_theta != NULL) {
        free(_theta);
        _theta = NULL;
    }
    
    if (_randomX != NULL) {
        free(_randomX);
        _randomX = NULL;
    }
    
    if (_randomY != NULL) {
        free(_randomY);
        _randomY = NULL;
    }
}

#pragma mark - SoftMax Main

- (void)randomPick:(int)maxSize
{
    long rNum = random();
    for (int i = 0; i < _randSize; i++) {
        _randomX[i] = _image[(rNum+i) % maxSize];
        _randomY[i] = _label[(rNum+i) % maxSize];
    }
}
/*
- (double *)MaxPro:(double *)index
{
    long double maxNum = index[0];
    for (int i = 1; i < _kType; i++) {
        maxNum = MAX(maxNum, index[i]);
    }
    
    long double sum = 0;
    for (int i = 0; i < _kType; i++) {
        index[i] -= maxNum;
        index[i] = expl(index[i]);
        sum += index[i];
    }
    
    for (int i = 0; i < _kType; i++) {
        index[i] /= sum;
    }
    return index;
}

- (void)updateModel:(double *)index currentPos:(int)pos
{
    double *delta = malloc(sizeof(double) * _kType);
    for (int i = 0; i < _kType; i++) {
        if (i != _randomY[pos]) {
            delta[i] = 0.0 - index[i];
        }
        else
        {
            delta[i] = 1.0 - index[i];
        }
        
        _bias[i] -= _descentRate * delta[i];
        
        for (int j = 0; j < _dim; j++) {
            _theta[i * _dim +j] += _descentRate * delta[i] * _randomX[pos][j] / _randSize;
        }
    }
    
    if (delta != NULL) {
        free(delta);
        delta = NULL;
    }
}

- (void)train
{
    _randomX = malloc(sizeof(double) * _randSize);
    _randomY = malloc(sizeof(int) * _randSize);
    double *index = malloc(sizeof(double) * _kType);
    
    for (int i = 0; i < _iterNum; i++) {
        [self randomPick:_trainNum];
        for (int j = 0; j < _randSize; j++) {
            // calculate wx+b
            vDSP_mmulD(_theta, 1, _randomX[j], 1, index, 1, _kType, 1, _dim);
            vDSP_vaddD(index, 1, _bias, 1, index, 1, _kType);
            
            index = [self MaxPro:index];
            [self updateModel:index currentPos:j];
        }
    }
    if (index != NULL) {
        free(index);
        index = NULL;
    }
}
*/

- (int)indicator:(int)label var:(int)x
{
    if (label == x) {
        return 1;
    }
    return 0;
}

- (double)sigmod:(int)type index:(int) index
{
    double up = 0;
    for (int i = 0; i < _dim; i++) {
        up += _theta[type * _dim + i] * _randomX[index][i];
    }
    up += _bias[type];
    
    double *down = malloc(sizeof(double) * _kType);
    double maxNum = -0xfffffff;
    vDSP_mmulD(_theta, 1, _randomX[index], 1, down, 1, _kType, 1, _dim);
    vDSP_vaddD(down, 1, _bias, 1, down, 1, _kType);
    
    for (int i = 0; i < _kType; i++) {
        maxNum = MAX(maxNum, down[i]);
    }
    
    double sum = 0;
    for (int i = 0; i < _kType; i++) {
        down[i] -= maxNum;
        sum += exp(down[i]);
    }
    
    if (down != NULL) {
        free(down);
        down = NULL;
    }
    
    return exp(up - maxNum) / sum;
}

- (double *)fderivative:(int)type
{
    double *outP = malloc(sizeof(double) * _dim);
    for (int i = 0; i < _dim; i++) {
        outP[i] = 0;
    }
    
    double *inner = malloc(sizeof(double) * _dim);
    for (int i = 0; i < _randSize; i++) {
        long double sig = [self sigmod:type index:i];
        int ind = [self indicator:_randomY[i] var:type];
        double loss = -_descentRate * (ind - sig) / _randSize;
        _bias[type] += loss * _randSize;
        vDSP_vsmulD(_randomX[i], 1, &loss, inner, 1, _dim);
        vDSP_vaddD(outP, 1, inner, 1, outP, 1, _dim);
    }
    if (inner != NULL) {
        free(inner);
        inner = NULL;
    }
    
    return outP;
}

- (void)train
{
    _randomX = malloc(sizeof(double) * _randSize);
    _randomY = malloc(sizeof(int) * _randSize);
    for (int i = 0; i < _iterNum; i++) {
        [self randomPick:_trainNum];
        for (int j = 0; j < _kType; j++) {
            double *newTheta = [self fderivative:j];
            for (int m = 0; m < _dim; m++) {
                _theta[j * _dim + m] = _theta[j * _dim + m] - _descentRate * newTheta[m];
            }
            if (newTheta != NULL) {
                free(newTheta);
                newTheta = NULL;
            }
        }
    }
}

- (void)saveTrainDataToDisk
{
    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSString *thetaPath = [[NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES) objectAtIndex:0] stringByAppendingString:@"/Theta.txt"];
//    NSLog(@"%@",thetaPath);
    NSData *data = [NSData dataWithBytes:_theta length:sizeof(double) *  _dim * _kType];
    [fileManager createFileAtPath:thetaPath contents:data attributes:nil];
    
    NSString *biasPath = [[NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES) objectAtIndex:0] stringByAppendingString:@"/bias.txt"];
    data = [NSData dataWithBytes:_bias length:sizeof(double) * _kType];
    [fileManager createFileAtPath:biasPath contents:data attributes:nil];
}

- (int)predict:(double *)image
{
    double maxNum = -0xffffff;
    int label = -1;
    double *index = malloc(sizeof(double) * _kType);
    vDSP_mmulD(_theta, 1, image, 1, index, 1, _kType, 1, _dim);
    vDSP_vaddD(index, 1, _bias, 1, index, 1, _kType);
    for (int i = 0; i < _kType; i++) {
        if (index[i] > maxNum) {
            maxNum = index[i];
            label = i;
        }
    }
    return label;
}

- (int)predict:(double *)image withOldTheta:(double *)theta andBias:(double *)bias
{
    double maxNum = -0xffffff;
    int label = -1;
    double *index = malloc(sizeof(double) * _kType);
    vDSP_mmulD(theta, 1, image, 1, index, 1, _kType, 1, _dim);
    vDSP_vaddD(index, 1, bias, 1, index, 1, _kType);
    for (int i = 0; i < _kType; i++) {
        if (index[i] > maxNum) {
            maxNum = index[i];
            label = i;
        }
    }
    return label;
}

@end
