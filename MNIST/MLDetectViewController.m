//
//  MLDetectViewController.m
//  MNIST
//
//  Created by Jiao Liu on 10/19/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import "MLDetectViewController.h"

@implementation MLDetectViewController

- (void)viewDidLoad
{
    [super viewDidLoad];
}

- (IBAction)BackBtnClicked:(id)sender {
    [self dismissViewControllerAnimated:YES completion:nil];
}

- (IBAction)CameraPick:(id)sender {
    UIImagePickerController *imgPick = [[UIImagePickerController alloc] init];
    imgPick.delegate = self;
    imgPick.allowsEditing = YES;
    imgPick.sourceType = [UIImagePickerController isSourceTypeAvailable:UIImagePickerControllerSourceTypeCamera] ? UIImagePickerControllerSourceTypeCamera : UIImagePickerControllerSourceTypeSavedPhotosAlbum;
    [self presentViewController:imgPick animated:YES completion:nil];
}

- (IBAction)PhotoPick:(id)sender {
    UIImagePickerController *imgPick = [[UIImagePickerController alloc] init];
    imgPick.delegate = self;
    imgPick.allowsEditing = YES;
    imgPick.sourceType = UIImagePickerControllerSourceTypePhotoLibrary;
    [self presentViewController:imgPick animated:YES completion:nil];
}

- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary<NSString *,id> *)info
{
    [picker dismissViewControllerAnimated:YES completion:^{
//        _presentedImage.image = [info objectForKey:UIImagePickerControllerEditedImage];
        [self predict:[info objectForKey:UIImagePickerControllerEditedImage]];
    }];
}

- (void)predict:(UIImage *)image
{
    UIImage *compressedImg;
    CGSize sz = CGSizeMake(28, 28);
    UIGraphicsBeginImageContext(sz);
    [image drawInRect:CGRectMake(0, 0, sz.width, sz.height)];
    compressedImg = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    double *imagePixels = [self getGrayImagePixel:compressedImg];
    _resultLabel.text = [NSString stringWithFormat:@"%d", [_softMax predict:imagePixels]];
//    _presentedImage.image = compressedImg;
    if (imagePixels != NULL) {
        free(imagePixels);
        imagePixels = NULL;
    }
}

- (double *)getGrayImagePixel:(UIImage *)image
{
    int width = image.size.width;
    int height = image.size.height;
    double *pixels = malloc(sizeof(double) * width * height);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();
    unsigned char *rawData = (unsigned char*) calloc(height * width, sizeof(unsigned char));
    NSUInteger bytesPerPixel = 1;
    NSUInteger bytesPerRow = bytesPerPixel * width;
    NSUInteger bitsPerComponent = 8;
    CGContextRef context = CGBitmapContextCreate(rawData, width, height,
                                                 
                                                 bitsPerComponent, bytesPerRow, colorSpace,
                                                 
                                                 kCGImageAlphaNone);
    
    CGColorSpaceRelease(colorSpace);
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), image.CGImage);
    _presentedImage.image = [UIImage imageWithCGImage:CGBitmapContextCreateImage(context)];
    CGContextRelease(context);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            int pixelInfo = (height * i) + j;
            pixels[i * height + j] = rawData[pixelInfo];
        }
    }
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            printf("%.1f ",pixels[i * height + j]);
        }
        printf("\n");
    }
    free(rawData);
    return pixels;
}

@end
