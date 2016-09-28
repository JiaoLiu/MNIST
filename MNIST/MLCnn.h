//
//  MLCnn.h
//  MNIST
//
//  Created by Jiao Liu on 9/28/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface MLCnn : NSObject

+ (double)truncated_normal:(double)mean dev:(double)stddev;

@end
