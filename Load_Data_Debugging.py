# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 19:18:19 2018

@author: Sonic
"""

import liga_data
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])
    print("***************************************")
    print("******            New Run         *****")
    print('\nDebug Loading')
    test_x, test_y = liga_data.load_datafromfile('Football-data-2018-1.csv')
    
    
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)