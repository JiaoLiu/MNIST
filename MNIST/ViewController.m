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
{
    MLSoftMax *softMax;
}

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
//    MLLoadMNIST *loader = [[MLLoadMNIST alloc] init];
//    NSArray *trainImage = [loader readImageData:@"/Users/Jiao/Desktop/MNIST/train-images-idx3-ubyte"];
//    NSArray *trainLabel = [loader readLabelData:@"/Users/Jiao/Desktop/MNIST/train-labels-idx1-ubyte"];
    
    UITapGestureRecognizer *tap = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(closeKeyboard)];
    [self.view addGestureRecognizer:tap];
}

- (void)closeKeyboard
{
    [self.view endEditing:YES];
}

- (IBAction)startTrain:(id)sender {
    [self closeKeyboard];
    _outputView.text = [[_outputView text] stringByAppendingString:@"start training!\n"];
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        NSString *bundlepPath = [[NSBundle mainBundle] resourcePath];
        double **trainImage = readImageData([[bundlepPath stringByAppendingString:@"/train-images-idx3-ubyte"] UTF8String]);
        int *trainLabel = readLabelData([[bundlepPath stringByAppendingString:@"/train-labels-idx1-ubyte"] UTF8String]);
        
        softMax = [[MLSoftMax alloc] initWithLoopNum:[_LoopNumText.text intValue] dim:28*28 type:10 size:[_LoopSizeText.text intValue] descentRate:[_DrateText.text doubleValue]];
        softMax.image = trainImage;
        softMax.label = trainLabel;
        softMax.trainNum = trainNum;
        [softMax train];
        
        [softMax saveTrainDataToDisk];
        
        dispatch_async(dispatch_get_main_queue(), ^{
            _outputView.text = [[_outputView text] stringByAppendingString:@"complete training!\n"];
        });
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
    });
}

- (IBAction)startCheckModel:(id)sender {
    [self closeKeyboard];
    if (!softMax) {
        UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Error" message:@"Please Train First" preferredStyle:UIAlertControllerStyleAlert];
        [alert addAction:[UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleCancel handler:nil]];
        [self presentViewController:alert animated:YES completion:nil];
        return;
    }
    
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        NSString *bundlepPath = [[NSBundle mainBundle] resourcePath];
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
        dispatch_async(dispatch_get_main_queue(), ^{
            _outputView.text = [[_outputView text] stringByAppendingFormat:@"Correct Ratio：%f%%\n",correct / 100.0];
        });
        printf("%f\n",correct / 10000.0);
/*
        if (!softMax) {
            softMax = [[MLSoftMax alloc] init];
            softMax.dim = 28 * 28;
            softMax.kType = 10;
        }
        
        NSFileManager *fileManager = [NSFileManager defaultManager];
        NSString *thetaPath = [[NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES) objectAtIndex:0] stringByAppendingString:@"/Theta.txt"];
        NSString *biasPath = [[NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES) objectAtIndex:0] stringByAppendingString:@"/bias.txt"];
        if ([fileManager fileExistsAtPath:biasPath] && [fileManager fileExistsAtPath:thetaPath] ) {
            double *theta = (double *)[[NSData dataWithContentsOfFile:thetaPath] bytes];
            double *bias = (double *)[[NSData dataWithContentsOfFile:biasPath] bytes];
            int correct = 0;
            for (int i = 0; i < testNum; i++) {
                int pred = [softMax predict:testImage[i] withOldTheta:theta andBias:bias];
                if (pred == testLabel[i]) {
                    correct++;
                }
            }
            dispatch_async(dispatch_get_main_queue(), ^{
                _outputView.text = [[_outputView text] stringByAppendingFormat:@"%f\n",correct / 10000.0];
            });
        }
        else
        {
            UIAlertController *alert = [UIAlertController alertControllerWithTitle:nil message:@"Please Train First" preferredStyle:UIAlertControllerStyleAlert];
            [self presentViewController:alert animated:YES completion:nil];
        }
        
        //    int correct = 0;
        //    for (int i = 0; i < testNum; i++) {
        //        int pred = [softMax predict:testImage[i]];
        //        if (pred == testLabel[i]) {
        //            correct++;
        //        }
        //        printf("%d - %d \n",testLabel[i],pred);
        //    }
        //    printf("%f\n",correct / 10000.0);
*/
        /* free memory */
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
    });
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
