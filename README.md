# 16spr_cao_gonzales_hoffman
Machine Learning 2016 Spring, Class project repo for Aaron Gonzales and Steven Hoffman

We have chosen a project with a *deep learning thrust* that will focus primarily on implementing the work done by Hong, et al. in [*Online Tracking by Learning Discriminative Saliency Map with Convolutional Neural Network*](http://jmlr.org/proceedings/papers/v37/hong15.pdf) from [ICML 2015](http://jmlr.org/proceedings/papers/v37/).

Note, this assumes that you have downloaded *and* compiled [MatConvNet](http://www.vlfeat.org/matconvnet/install/) and [OnlineSVM Code](http://www.isn.ucsd.edu/svm/incremental/) into directories named *matconvnet/* and *onlinesvm/* respectively, and that you have the appropriate [data](https://sites.google.com/site/trackerbenchmark/benchmarks/v10) in a directory named *data/*, as well as a directory named *net/* to contain downloaded CNNs, all under the project's root directory.

To run the code, first open up *track_object.m*. The first section of this file has a list of parameters that may be set/changed in order to change the video (test_video, test_video_dir), the ground truth bounding box file (test_video_gt), where the outputs should be saved (bb_out_filename, result_out_dir), and the maximum number of frames to track the object for (max_num_frames), among several others. After setting these parameters, run *track_object.m* to perform the object tracking.
