//
//  MLDetectViewController.h
//  MNIST
//
//  Created by Jiao Liu on 10/19/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import <UIKit/UIKit.h>
#import "MLSoftMax.h"

@interface MLDetectViewController : UIViewController<UIImagePickerControllerDelegate,UINavigationControllerDelegate>
@property (weak, nonatomic) IBOutlet UIImageView *presentedImage;
@property (weak, nonatomic) IBOutlet UILabel *resultLabel;
@property (weak, nonatomic) MLSoftMax *softMax;

@end
