//
//  ViewController.m
//  MNIST
//
//  Created by Jiao Liu on 9/23/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import "ViewController.h"
#import "MLLoadMNIST.h"
#import <Accelerate/Accelerate.h>
#import "MLSoftMax.h"

static const int trainNum = 60000;
static const int testNum = 10000;

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
//    float mat1[3][3][1][2] = {{{
//        {1,2,3},
//        {2,3,4},
//        {4,5,6}},
//        {
//            {0,0,0},
//            {0,0,0},
//            {1,1,1}}}};
//    float mat2[3] = {1,2,3};
//    float res[3] = {};
//    cblas_sgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0f, (float *)mat1[0][1], 3, mat2, 1, 1.0f, res, 1);
//    vDSP_mmul(*mat1[0][1], 1, mat2, 1, res, 1, 3, 1, 3);
//    
//    
//    float inVector1[8] = {1, 2, 3, 4, 5, 6, 7, 8};
//    float inVector2[8] = {1, 2, 3, 4, 5, 6, 7, 8};
//    float outVector[8];
//    
//    vDSP_vadd(inVector1, 1, inVector2, 1, outVector, 1, 8);
    
    
//    MLLoadMNIST *loader = [[MLLoadMNIST alloc] init];
//    NSArray *trainImage = [loader readImageData:@"/Users/Jiao/Desktop/MNIST/train-images-idx3-ubyte"];
//    NSArray *trainLabel = [loader readLabelData:@"/Users/Jiao/Desktop/MNIST/train-labels-idx1-ubyte"];
    
    [self startCalculation];
}

- (void)startCalculation
{
    NSString *bundlepPath = [[NSBundle mainBundle] resourcePath];
    double **trainImage = readImageData([[bundlepPath stringByAppendingString:@"/train-images-idx3-ubyte"] UTF8String]);
    int *trainLabel = readLabelData([[bundlepPath stringByAppendingString:@"/train-labels-idx1-ubyte"] UTF8String]);
    
    MLSoftMax *softMax = [[MLSoftMax alloc] initWithLoopNum:500 dim:28*28 type:10 size:100];
    softMax.image = trainImage;
    softMax.label = trainLabel;
    softMax.trainNum = trainNum;
    softMax.descentRate = 0.01;
    [softMax train];
    
    double **testImage = readImageData([[bundlepPath stringByAppendingString:@"/t10k-images-idx3-ubyte"] UTF8String]);
    int *testLabel = readLabelData([[bundlepPath stringByAppendingString:@"/t10k-labels-idx1-ubyte"] UTF8String]);
    
    int correct = 0;
    for (int i = 0; i < testNum; i++) {
        int pred = [softMax predict:testImage[i]];
        if (pred == testLabel[i]) {
            correct++;
        }
        printf("%d - %d \n",testLabel[i],pred);
    }
    printf("%f",correct / 10000.0);

    /* free memory */
    if (trainImage != NULL) {
        for (int i = 0; i < trainNum; i++) {
            free(trainImage[i]);
            trainImage[i] = NULL;
        }
        free(trainImage);
        trainImage = NULL;
    }
    if (trainLabel != NULL) {
        free(trainLabel);
        trainLabel = NULL;
    }
    
    if (testImage != NULL) {
        for (int i = 0; i < testNum; i++) {
            free(testImage[i]);
            testImage[i] = NULL;
        }
        free(testImage);
        testImage = NULL;
    }
    if (testLabel != NULL) {
        free(testLabel);
        testLabel = NULL;
    }
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
