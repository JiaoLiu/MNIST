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

+ (double *)relu:(double *)x size:(int)size
{
    double *zero = [MLCnn fillVector:0.0f size:size];
    vDSP_vmaxD(x, 1, zero, 1, x, 1, size);
    if (zero != NULL) {
        free(zero);
        zero = NULL;
    }
    return x;
}

+ (double *)fillVector:(double)num size:(int)size
{
    double *outP = malloc(sizeof(double) * size);
    vDSP_vfillD(&num, outP, 1, size);
    return outP;

}

+ (double)max_pool:(double *)input dim:(int)dim row:(int)row col:(int)col stride:(NSArray *)stride
{
    double maxV = input[dim * [stride[0] intValue] + row * 2 * [stride[1] intValue] + col * 2];
    maxV = MAX(maxV, input[dim * [stride[0] intValue] + (row * 2 + 1) * [stride[1] intValue] + col * 2]);
    maxV = MAX(maxV, input[dim * [stride[0] intValue] + row * 2 * [stride[1] intValue] + col * 2 + 1]);
    maxV = MAX(maxV, input[dim * [stride[0] intValue] + (row * 2 + 1) * [stride[1] intValue] + col * 2 + 1]);
    return maxV;
}

+ (double)conv_2d:(double *)input filter:(double *)filter row:(int)row col:(int)col currentX:(int)x currentY:(int)y stride:(NSArray *)stride
{
    double sum = 0;
    int strideRow = row - [stride[0] intValue] * 2;
    int strideCol = col - [stride[1] intValue] * 2;
    for (int i = x; i < x + strideRow; i++) {
        for (int j = y; j < y + strideCol; j++) {
            sum += input[i * col + j] * filter[(i - x + [stride[0] intValue]) * strideCol + j - y + [stride[1] intValue]];
        }
    }
    return sum;
}

+ (double *)weight_init:(int)size
{
    double *outP = malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++) {
        outP[i] = [MLCnn truncated_normal:0 dev:0.1];
    }
    return outP;
}

+ (double *)bias_init:(int)size
{
    return [MLCnn fillVector:0.1f size:size];
}

# pragma mark - CNN Main

- (id)initWithFilters:(NSArray *)filters fullConnectSize:(int)size row:(int)dimRow col:(int)dimCol
{
    self = [super init];
    if (self) {
        _filters = filters;
        _connectSize = size;
        _numOfFilter = (int)[filters count];
        _dimRow = dimRow;
        _dimCol = dimCol;
        _weight = malloc(sizeof(double) * (_numOfFilter + 1));
        _bias = malloc(sizeof(double) * (_numOfFilter + 1));
        _filteredImage = malloc(sizeof(double) * (_numOfFilter + 1));
        int preDim = 1;
        int row = dimRow;
        int col = dimCol;
        for (int i = 0; i < _numOfFilter; i++) {
            _weight[i] = [MLCnn weight_init:[_filters[i][0] intValue] * [_filters[i][1] intValue] * [_filters[i][2] intValue] * preDim];
            _bias[i] = [MLCnn bias_init:[_filters[i][2] intValue]];
            row /= 2;
            col /= 2;
            preDim = [_filters[i][2] intValue];
            _filteredImage[i] = NULL;
        }
        _weight[_numOfFilter] = [MLCnn weight_init:row * col * preDim * _connectSize];
        _bias[_numOfFilter] = [MLCnn bias_init:_connectSize];
        _filteredImage[_numOfFilter] = NULL;
    }
    return self;
}

- (void)dealloc
{
    if (_weight != NULL) {
        for (int i = 0; i < _numOfFilter + 1; i++) {
            free(_weight[i]);
            _weight[i] = NULL;
        }
        free(_weight);
        _weight = NULL;
    }
    if (_bias != NULL) {
        for (int i = 0; i < _numOfFilter + 1; i++) {
            free(_bias[i]);
            _bias[i] = NULL;
        }
        free(_bias);
        _bias = NULL;
    }
    if (_filteredImage != NULL) {
        for (int i = 1; i < _numOfFilter + 1; i++) {
            free(_filteredImage[i]);
            _filteredImage[i] = NULL;
        }
        free(_filteredImage);
        _filteredImage = NULL;
    }
}

- (double *)filterImage:(double *)image
{
    if (_numOfFilter == 0) {
        return image;
    }
    
    int preDim = 1;
    int row = _dimRow;
    int col = _dimCol;
    _filteredImage[0] = image;
    for (int i = 0; i < _numOfFilter; i++) {
        double *conv = malloc(sizeof(double) * row * col * [_filters[i][2] intValue]);
        double fillNum = 0;
        vDSP_vfillD(&fillNum, conv, 1, row * col * [_filters[i][2] intValue]);
        if (_filteredImage[i+1] != NULL) {
            free(_filteredImage[i+1]);
            _filteredImage[i+1] = NULL;
        }
        _filteredImage[i+1] = malloc(sizeof(double) * row * col * [_filters[i][2] intValue] / 4);
        
        // convolve
        for (int k = 0; k < [_filters[i][2] intValue]; k++) {
            double *inner = malloc(sizeof(double) * row * col);
            for (int m = 0; m < preDim; m++) {
                vDSP_imgfirD((_filteredImage[i] + m * row * col), row, col, (_weight[i] + k * [_filters[i][0] intValue] * [_filters[i][1] intValue] * preDim + m * [_filters[i][0] intValue] * [_filters[i][1] intValue]), inner, [_filters[i][0] intValue], [_filters[i][1] intValue]);
                vDSP_vaddD((conv + k * row * col), 1, inner, 1, (conv + k * row * col), 1, row * col);
            }
            vDSP_vsaddD((conv + k * row * col), 1, &_bias[i][k], (conv + k * row * col), 1, row * col);
            if (inner != NULL) {
                free(inner);
                inner = NULL;
            }
        }
        conv = [MLCnn relu:conv size:row * col * [_filters[i][2] intValue]];
        
        // max pooling 2*2
        for (int k = 0; k < [_filters[i][2] intValue]; k++) {
            for (int m = 0; m < row / 2; m++) {
                for (int n = 0; n < col / 2; n++) {
                    _filteredImage[i+1][k * row * col / 4 + m * col / 2 + n] = [MLCnn max_pool:conv dim:k row:m col:n stride:@[[NSNumber numberWithInt:row * col],[NSNumber numberWithInt:col]]];
                }
            }
        }
        
        row /= 2;
        col /= 2;
        preDim = [_filters[i][2] intValue];

        if (conv != NULL) {
            free(conv);
            conv = NULL;
        }
    }
    
    double *outP = malloc(sizeof(double) * _connectSize);
    vDSP_mmulD(_weight[_numOfFilter], 1, _filteredImage[_numOfFilter], 1, outP, 1, _connectSize, 1, row * col * preDim);
    vDSP_vaddD(outP, 1, _bias[_numOfFilter], 1, outP, 1, _connectSize);
    outP = [MLCnn relu:outP size:_connectSize];
    return outP;
}

- (void)backPropagation:(double *)loss
{
    int row = _dimRow / pow(2, _numOfFilter);
    int col = _dimCol /  pow(2, _numOfFilter);
    // update full-connect layer
    vDSP_vaddD(loss, 1, _bias[_numOfFilter], 1, _bias[_numOfFilter], 1, _connectSize);
    double *flayerLoss = malloc(sizeof(double) * row * col * [_filters[_numOfFilter - 1][2] intValue]);
    double *transWeight = malloc(sizeof(double) * row * col * [_filters[_numOfFilter - 1][2] intValue] * _connectSize);
    vDSP_mtransD(_weight[_numOfFilter], 1, transWeight, 1, row * col * [_filters[_numOfFilter - 1][2] intValue], _connectSize);
    vDSP_mmulD(transWeight, 1, loss, 1, flayerLoss, 1, row * col * [_filters[_numOfFilter - 1][2] intValue], 1, _connectSize);
    
    double *flayerWeight = malloc(sizeof(double) * row * col * [_filters[_numOfFilter - 1][2] intValue] * _connectSize);
    vDSP_mmulD(loss, 1, _filteredImage[_numOfFilter], 1, flayerWeight, 1, _connectSize, row * col * [_filters[_numOfFilter - 1][2] intValue], 1);
    vDSP_vaddD(_weight[_numOfFilter], 1, flayerWeight, 1, _weight[_numOfFilter], 1, row * col * [_filters[_numOfFilter - 1][2] intValue] * _connectSize);
    
    if (loss != NULL) {
        free(loss);
        loss = NULL;
    }
    if (flayerWeight != NULL) {
        free(flayerWeight);
        flayerWeight = NULL;
    }
    if (transWeight != NULL) {
        free(transWeight);
        transWeight = NULL;
    }
  
    // update Conv & pooling layer
    double *convBackLoss = flayerLoss;
    for (int i = _numOfFilter - 1; i >= 0; i--) {
        // unsampling
        row *= 2;
        col *= 2;
        int preDim = i > 0 ? [_filters[i-1][2] intValue] : 1;
        double *unsample = malloc(sizeof(double) * row * col * [_filters[i][2] intValue]);
        for (int k = 0; k < [_filters[i][2] intValue]; k++) {
            for (int m = 0; m < row / 2; m++) {
                for (int n = 0; n < col / 2; n++) {
                    unsample[k*row*col + m*2*col + n*2] = unsample[k*row*col + m*2*col + n*2 + 1] = unsample[k*row*col + (m*2+1)*col + n*2] = unsample[k*row*col + (m*2+1)*col + n*2 + 1] = convBackLoss[k*row*col/4 + m*col/2 + n];
                }
            }
        }
        
        // update conv bias
        for (int k = 0; k < [_filters[i][2] intValue]; k++) {
            double biasLoss = 0;
            for (int m = 0; m < row / 2; m++) {
                for (int n = 0; n < col / 2; n++) {
                    biasLoss += convBackLoss[k*row*col/4 + m*col/2 + n] * 4;
                }
            }
            _bias[i][k] += biasLoss;
        }

        if (i > 0) { //if not the first layer calculate back loss
            if (convBackLoss != NULL) {
                free(convBackLoss);
                convBackLoss = NULL;
            }
            convBackLoss = [MLCnn fillVector:0.0f size:row * col * preDim];
            
            // Δq′=(∑p∈CΔp∗frot180(Θp))∘ϕ′(Oq′)
            for (int k = 0; k < preDim; k++) {
                double *inner = malloc(sizeof(double) * row * col);
                for (int m = 0; m < [_filters[i][2] intValue]; m++) {
                    double *reverseWeight = [MLCnn fillVector:0.0f size:[_filters[i][0] intValue] * [_filters[i][1] intValue]];
                    vDSP_vaddD(reverseWeight, 1, (_weight[i] + m * [_filters[i][0] intValue] * [_filters[i][1] intValue] * preDim + k * [_filters[i][0] intValue] * [_filters[i][1] intValue]), 1, reverseWeight, 1, [_filters[i][0] intValue] * [_filters[i][1] intValue]);
                    vDSP_vrvrsD(reverseWeight, 1, [_filters[i][0] intValue] * [_filters[i][1] intValue]);
                    vDSP_imgfirD((unsample + m * row * col), row, col, reverseWeight, inner, [_filters[i][0] intValue], [_filters[i][1] intValue]);
                    vDSP_vaddD((convBackLoss + k * row * col), 1, inner, 1, (convBackLoss + k * row * col), 1, row * col);
                    if (reverseWeight != NULL) {
                        free(reverseWeight);
                        reverseWeight = NULL;
                    }
                }
                if (inner != NULL) {
                    free(inner);
                    inner = NULL;
                }
            }
        }

        // update conv weight
        for (int k = 0; k < [_filters[i][2] intValue]; k++) {
            int strideRow = [_filters[i][0] intValue] / 2;
            int strideCol = [_filters[i][1] intValue] / 2;
            double *curLoss = malloc(sizeof(double) * (row - strideRow * 2) * (col - strideCol * 2));
            for (int p = 0; p < row - strideRow * 2; p++) {
                for (int q = 0; q < col - strideCol * 2; q++) {
                    curLoss[p * (col - strideCol * 2) + q] = unsample[k * row * col + (p + strideRow) * col + q + strideCol];
                }
            }
            vDSP_vrvrsD(curLoss, 1, (row - strideRow * 2) * (col - strideCol * 2));
            
            for (int m = 0; m < preDim; m++) {
                double *inner = malloc(sizeof(double) * row * col);
                vDSP_imgfirD((_filteredImage[i] + m * row * col), row, col, curLoss, inner, row - strideRow * 2, col - strideCol * 2);
                double *weightLoss = malloc(sizeof(double) * [_filters[i][0] intValue] * [_filters[i][1] intValue]);
                int P = (row - strideRow * 2) / 2;
                int Q = (col - strideCol * 2) / 2;
                for (int r = P; r <= row - P; ++r)
                {
                    for (int c = Q; c <= col - Q; ++c)
                    {
                        weightLoss[(r-P)*[_filters[i][1] intValue] + (c-Q)] = inner[r*col + c];
                    }
                }
                vDSP_vrvrsD(weightLoss, 1, [_filters[i][0] intValue] * [_filters[i][1] intValue]);
//                double divDim = preDim;
//                vDSP_vsdivD(weightLoss, 1, &divDim, weightLoss, 1, [_filters[i][0] intValue] * [_filters[i][1] intValue]);
                vDSP_vaddD((_weight[i] + k * [_filters[i][0] intValue] * [_filters[i][1] intValue] * preDim + m * [_filters[i][0] intValue] * [_filters[i][1] intValue]), 1, weightLoss, 1, (_weight[i] + k * [_filters[i][0] intValue] * [_filters[i][1] intValue] * preDim + m * [_filters[i][0] intValue] * [_filters[i][1] intValue]), 1, [_filters[i][0] intValue] * [_filters[i][1] intValue]);
                
                if (weightLoss != NULL) {
                    free(weightLoss);
                    weightLoss = NULL;
                }
                if (inner != NULL) {
                    free(inner);
                    inner = NULL;
                }
            }
            if (curLoss != NULL) {
                free(curLoss);
                curLoss = NULL;
            }
            
            
//            for (int m = 0; m < preDim; m++) {
//                double *inner = malloc(sizeof(double) * [_filters[i][0] intValue] * [_filters[i][1] intValue]);
//                for (int p = 0; p < [_filters[i][0] intValue]; p++) {
//                    for (int q = 0; q < [_filters[i][1] intValue]; q++) {
//                        inner[p * [_filters[i][1] intValue] + q] = [MLCnn conv_2d:(_filteredImage[i] + m * row * col) filter:(unsample + k * row * col) row:row col:col currentX:p currentY:q stride:@[[NSNumber numberWithInt:[_filters[i][0] intValue] / 2],[NSNumber numberWithInt:[_filters[i][1] intValue] / 2]]];
//                    }
//                }
//                vDSP_vrvrsD(inner, 1, [_filters[i][0] intValue] * [_filters[i][1] intValue]);
//                vDSP_vaddD((_weight[i] + k * [_filters[i][0] intValue] * [_filters[i][1] intValue] * preDim + m * [_filters[i][0] intValue] * [_filters[i][1] intValue]), 1, inner, 1, (_weight[i] + k * [_filters[i][0] intValue] * [_filters[i][1] intValue] * preDim + m * [_filters[i][0] intValue] * [_filters[i][1] intValue]), 1, [_filters[i][0] intValue] * [_filters[i][1] intValue]);
//                
//                
//                if (inner != NULL) {
//                    free(inner);
//                    inner = NULL;
//                }
//            }
        }

        if (unsample != NULL) {
            free(unsample);
            unsample = NULL;
        }
    }
 
    if (convBackLoss != NULL) {
        free(convBackLoss);
        convBackLoss = NULL;
    }
    
    for (int k = 0; k < 10; k++) {
        printf("%f ",_weight[0][k]);
    }
    printf("\n");
}

@end
