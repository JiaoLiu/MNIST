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

+ (double)mean_pool:(double *)input dim:(int)dim row:(int)row col:(int)col stride:(NSArray *)stride
{
    double sum = input[dim * [stride[0] intValue] + row * 2 * [stride[1] intValue] + col * 2];
    sum += input[dim * [stride[0] intValue] + (row * 2 + 1) * [stride[1] intValue] + col * 2];
    sum += input[dim * [stride[0] intValue] + row * 2 * [stride[1] intValue] + col * 2 + 1];
    sum += input[dim * [stride[0] intValue] + (row * 2 + 1) * [stride[1] intValue] + col * 2 + 1];
    return sum / 4;
}

+ (void)conv_2d:(double *)input inputRow:(int)NR inputCol:(int)NC filter:(double *)filter output:(double *)output filterRow:(int)P filterCol:(int)Q
{
    int outRow = NR - P + 1;
    int outCol = NR - Q + 1;
    for (int i = 0; i < outRow; i++) {
        for (int j = 0; j < outCol; j++) {
            double sum = 0;
            for (int k = 0; k < P; k++) {
                double *inner = malloc(sizeof(double) * Q);
                vDSP_vmulD((input + (i + k) * NR + j), 1, (filter + k * Q), 1, inner, 1, Q);
                vDSP_vswsumD(inner, 1, &sum, 1, 1, Q);
                if (inner != NULL) {
                    free(inner);
                    inner = NULL;
                }
            }
            output[i* outCol + j] = sum;
        }
    }
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

- (id)initWithFilters:(NSArray *)filters fullConnectSize:(int)size row:(int)dimRow col:(int)dimCol keepRate:(double)rate
{
    self = [super init];
    if (self) {
        _filters = filters;
        _connectSize = size;
        _numOfFilter = (int)[filters count];
        _dimRow = dimRow;
        _dimCol = dimCol;
        _keepProb = rate;
        _weight = malloc(sizeof(double) * (_numOfFilter + 1));
        _bias = malloc(sizeof(double) * (_numOfFilter + 1));
        _filteredImage = malloc(sizeof(double) * (_numOfFilter + 1));
        _reluFlag = malloc(sizeof(double) * (_numOfFilter + 1));
        _dropoutMask = malloc(sizeof(double) * (_connectSize));
        int preDim = 1;
        int row = dimRow;
        int col = dimCol;
        for (int i = 0; i < _numOfFilter; i++) {
            _weight[i] = [MLCnn weight_init:[_filters[i][0] intValue] * [_filters[i][1] intValue] * [_filters[i][2] intValue] * preDim];
            _bias[i] = [MLCnn bias_init:[_filters[i][2] intValue]];
            row = (row - ([_filters[i][0] intValue] / 2) * 2) / 2;
            col = (col - ([_filters[i][1] intValue] / 2) * 2) / 2;
            preDim = [_filters[i][2] intValue];
            _filteredImage[i] = NULL;
            _reluFlag[i] = NULL;
        }
        _weight[_numOfFilter] = [MLCnn weight_init:row * col * preDim * _connectSize];
        _bias[_numOfFilter] = [MLCnn bias_init:_connectSize];
        _filteredImage[_numOfFilter] = NULL;
        _reluFlag[_numOfFilter] = NULL;
        _outRow = row;
        _outCol = col;
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
    if (_reluFlag != NULL) {
        for (int i = 0; i < _numOfFilter + 1; i++) {
            free(_reluFlag[i]);
            _reluFlag[i] = NULL;
        }
        free(_reluFlag);
        _reluFlag = NULL;
    }
    if (_dropoutMask != NULL) {
        free(_dropoutMask);
        _dropoutMask = NULL;
    }
}

- (double *)filterImage:(double *)image state:(BOOL)isTraining
{
    if (_numOfFilter == 0) {
        return image;
    }
    
    int preDim = 1;
    int row = _dimRow;
    int col = _dimCol;
    _filteredImage[0] = image;
    for (int i = 0; i < _numOfFilter; i++) {
        double *conv = [MLCnn fillVector:0.0f size:row * col * [_filters[i][2] intValue]];
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
        
        int strideRow = [_filters[i][0] intValue] / 2;
        int strideCol = [_filters[i][1] intValue] / 2;
        row -= strideRow * 2;
        col -= strideCol * 2;
        if (_reluFlag[i] != NULL) {
            free(_reluFlag[i]);
            _reluFlag[i] = NULL;
        }
        _reluFlag[i] = malloc(sizeof(double) * row * col * [_filters[i][2] intValue]);
        for (int k = 0; k < [_filters[i][2] intValue]; k++) {
            for (int r = 0; r < row; ++r)
            {
                for (int c = 0; c < col; ++c)
                {
                    _reluFlag[i][k * row *col + r * col + c] = conv[k * (row + strideRow * 2) * (col + strideCol * 2) + (r + strideRow) * (col + strideCol * 2) + c + strideCol];
                }
                
            }
        }
        // relu
        _reluFlag[i] = [MLCnn relu:_reluFlag[i] size:row * col * [_filters[i][2] intValue]];
        
        // pooling 2*2
        if (_filteredImage[i+1] != NULL) {
            free(_filteredImage[i+1]);
            _filteredImage[i+1] = NULL;
        }
        _filteredImage[i+1] = malloc(sizeof(double) * row * col * [_filters[i][2] intValue] / 4);
        
        for (int k = 0; k < [_filters[i][2] intValue]; k++) {
            for (int m = 0; m < row / 2; m++) {
                for (int n = 0; n < col / 2; n++) {
                    _filteredImage[i+1][k * row * col / 4 + m * col / 2 + n] = [MLCnn mean_pool:_reluFlag[i] dim:k row:m col:n stride:@[[NSNumber numberWithInt:row * col],[NSNumber numberWithInt:col]]];
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
    
    // full connect
    if (_reluFlag[_numOfFilter] != NULL) {
        free(_reluFlag[_numOfFilter]);
        _reluFlag[_numOfFilter] = NULL;
    }
    _reluFlag[_numOfFilter] = malloc(sizeof(double) * _connectSize);
    vDSP_mmulD(_weight[_numOfFilter], 1, _filteredImage[_numOfFilter], 1, _reluFlag[_numOfFilter], 1, _connectSize, 1, row * col * preDim);
    vDSP_vaddD(_reluFlag[_numOfFilter], 1, _bias[_numOfFilter], 1, _reluFlag[_numOfFilter], 1, _connectSize);
    _reluFlag[_numOfFilter] = [MLCnn relu:_reluFlag[_numOfFilter] size:_connectSize];
    
    // dropOut
    if (isTraining) {
        for (int i = 0; i < _connectSize; i++) {
            if ((double)rand()/RAND_MAX > _keepProb) {
                _dropoutMask[i] = 0;
                _reluFlag[_numOfFilter][i] = 0;
            }
            else
            {
                _dropoutMask[i] = 1;
            }
        }
    }
    else
    {
        vDSP_vsmulD(_reluFlag[_numOfFilter], 1, &_keepProb, _reluFlag[_numOfFilter], 1, _connectSize);
    }
    
    return _reluFlag[_numOfFilter];
}

- (void)backPropagation:(double *)loss
{
    int row = _outRow;
    int col = _outCol;
    // dropOut
    vDSP_vmulD(loss, 1, _dropoutMask, 1, loss, 1, _connectSize);
    
    // deRelu
    for (int i = 0; i < _connectSize; i++) {
        if (_reluFlag[_numOfFilter][i] == 0) {
            loss[i] = 0;
        }
    }
    
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
                    unsample[k*row*col + m*2*col + n*2] = unsample[k*row*col + m*2*col + n*2 + 1] = unsample[k*row*col + (m*2+1)*col + n*2] = unsample[k*row*col + (m*2+1)*col + n*2 + 1] = convBackLoss[k*row*col/4 + m*col/2 + n] / 4;
                }
            }
        }
        // deRelu
        for (int k = 0; k < row * col * [_filters[i][2] intValue]; k++) {
            if (_reluFlag[i][k] == 0) {
                unsample[k] = 0;
            }
        }

        // update conv bias
        for (int k = 0; k < [_filters[i][2] intValue]; k++) {
            double biasLoss = 0;
            for (int m = 0; m < row / 2; m++) {
                for (int n = 0; n < col / 2; n++) {
                    biasLoss += convBackLoss[k*row*col/4 + m*col/2 + n];
                }
            }
            _bias[i][k] += biasLoss;
        }
        
        int strideRow = [_filters[i][0] intValue] / 2;
        int strideCol = [_filters[i][1] intValue] / 2;

        if (i > 0) { //if not the first layer calculate back loss
            if (convBackLoss != NULL) {
                free(convBackLoss);
                convBackLoss = NULL;
            }
            convBackLoss = [MLCnn fillVector:0.0f size:(row + strideRow * 2) * (col + strideCol * 2) * preDim];
            double *curLoss = [MLCnn fillVector:0.0f size:(row + strideRow * 2) * (col + strideCol * 2) * [_filters[i][2] intValue]];
            for (int k = 0; k < [_filters[i][2] intValue]; k++) {
                for (int p = 0; p < row; p++) {
                    for (int q = 0; q < col; q++) {
                        curLoss[k * (row + strideRow * 2) * (col + strideCol * 2) + (p + strideRow) * (col + strideCol * 2) + q + strideCol] = unsample[k * row * col + p * col + q];
                    }
                }
            }
            
            // Δq′=(∑p∈CΔp∗frot180(Θp))∘ϕ′(Oq′)
            for (int k = 0; k < preDim; k++) {
                double *inner = malloc(sizeof(double) * (row + strideRow * 2) * (col + strideCol * 2));
                for (int m = 0; m < [_filters[i][2] intValue]; m++) {
                    double *reverseWeight = [MLCnn fillVector:0.0f size:[_filters[i][0] intValue] * [_filters[i][1] intValue]];
                    vDSP_vaddD(reverseWeight, 1, (_weight[i] + m * [_filters[i][0] intValue] * [_filters[i][1] intValue] * preDim + k * [_filters[i][0] intValue] * [_filters[i][1] intValue]), 1, reverseWeight, 1, [_filters[i][0] intValue] * [_filters[i][1] intValue]);
                    vDSP_vrvrsD(reverseWeight, 1, [_filters[i][0] intValue] * [_filters[i][1] intValue]);
                    vDSP_imgfirD((curLoss + m * (row + strideRow * 2) * (col + strideCol * 2)), row + strideRow * 2, col + strideCol * 2, reverseWeight, inner, [_filters[i][0] intValue], [_filters[i][1] intValue]);
                    vDSP_vaddD((convBackLoss + k * (row + strideRow * 2) * (col + strideCol * 2)), 1, inner, 1, (convBackLoss + k * (row + strideRow * 2) * (col + strideCol * 2)), 1, (row + strideRow * 2) * (col + strideCol * 2));
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
            if (curLoss != NULL) {
                free(curLoss);
                curLoss = NULL;
            }
        }

        // update conv weight
        for (int k = 0; k < [_filters[i][2] intValue]; k++) {
//            int strideRow = [_filters[i][0] intValue] / 2;
//            int strideCol = [_filters[i][1] intValue] / 2;
//            double *curLoss = malloc(sizeof(double) * (row - strideRow * 2) * (col - strideCol * 2));
//            for (int p = 0; p < row - strideRow * 2; p++) {
//                for (int q = 0; q < col - strideCol * 2; q++) {
//                    curLoss[p * (col - strideCol * 2) + q] = unsample[k * row * col + (p + strideRow) * col + q + strideCol];
//                }
//            }
//            vDSP_vrvrsD(curLoss, 1, (row - strideRow * 2) * (col - strideCol * 2));
            vDSP_vrvrsD((unsample + k * row * col), 1, row * col);
            
            for (int m = 0; m < preDim; m++) {
                double *inner = malloc(sizeof(double) * (row + strideRow * 2) * (col + strideCol * 2));
                vDSP_imgfirD((_filteredImage[i] + m * (row + strideRow * 2) * (col + strideCol * 2)), (row + strideRow * 2), (col + strideCol * 2), (unsample + k * row * col), inner, row, col);
                double *weightLoss = malloc(sizeof(double) * [_filters[i][0] intValue] * [_filters[i][1] intValue]);
                int P = row / 2;
                int Q = col / 2;
                for (int r = P; r <= (row + strideRow * 2) - P; ++r)
                {
                    for (int c = Q; c <= (col + strideCol * 2) - Q; ++c)
                    {
                        weightLoss[(r-P)*[_filters[i][1] intValue] + (c-Q)] = inner[r*col + c];
                    }
                }
//                [MLCnn conv_2d:(_filteredImage[i] + m * (row + strideRow * 2) * (col + strideCol * 2)) inputRow:(row + strideRow * 2) inputCol:(col + strideCol * 2) filter:(unsample + k * row * col) output:weightLoss filterRow:row filterCol:col];
                vDSP_vrvrsD(weightLoss, 1, [_filters[i][0] intValue] * [_filters[i][1] intValue]);
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
        }

        row += strideRow * 2;
        col += strideCol * 2;
        if (unsample != NULL) {
            free(unsample);
            unsample = NULL;
        }
         
    }
 
    if (convBackLoss != NULL) {
        free(convBackLoss);
        convBackLoss = NULL;
    }
    
//    for (int k = 0; k < 10; k++) {
//        printf("%f ",_weight[2][k]);
//    }
//    printf("\n");
}

@end
