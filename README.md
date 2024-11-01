# Vop2el

The goal of this repository is to provide a stereo visual odometry algorithm based on the  methods described in [SOFT2](https://lamor.fer.hr/images/50036607/2022-cvisic-soft2-tro.pdf) paper.

Note: This algorithm relies on a known camera calibration matrix and a known extrinsic transformation between the left and right cameras of the stereo camera.

![alt text](doc/result_sequence_00_kitti.gif)

## Dependencies

This project has been tested and verified to build successfully on Ubuntu 20.04 LTS with the following dependencies.

- Eigen 3.4

- Ceres 2.2

- OpenMP

- For OpenCV, we use a slightly modified version that will be built during the build of this project.

## Build and install

To build and install this project on linux, follow the steps below:

```bash
git clone https://github.com/Vop2elToolkit/Vop2el.git

cd Vop2el && git submodule update --init --recursive

mkdir ../Vop2el-build && cd ../Vop2el-build
```

#### Notes:
- ```vop2el``` has two build modes: with and without Rerun SDK 
- When building without Rerun SDK, users will receive a ```csv``` file containing the estimated poses after processing a sequence of images.
- When building with Rerun SDK, users will get a real time visualization of estimated poses, and receive a ```csv``` file containing the final poses after processing a sequence of images

### Build without Rerun SDK

```bash
cmake ../Vop2el -DBUILD_WITH_RERUN=OFF

cmake --build .
```    

### Build with Rerun SDK for visualization

Download the Rerun viewer (version 0.17.0) from https://github.com/rerun-io/rerun/releases

```bash
cmake ../Vop2el

cmake --build .
```

## Run

To run this project on linux, follow the steps below (Skip the initial two steps when building without Rerun):

- Launch the Rerun viewer

- In the Rerun viewer, open the blueprint file ```vop2el/vop2el_src/test/vop2el.rbl```

- Run Vop2el

```bash
./bin/Vop2elTester /path/to/left/images/folder \
                   /path/to/right/images/folder \
                   /path/to/config/ini/file \
                   /path/to/estimated/poses/text/file \
                   /path/to/ground/truth/poses/text/file(optional)
```


![alt text](doc/rerun_sequence_00_kitti.gif)


### Notes:
- Users should rely on parameters ini file provided in test folder.
- Using ground plane adds additional overhead to processing time and requires precise values for normal vector and distance to ground plane, users should try running the program without it first and then test with it if the results are not satisfactory.  
- To improve processing time, the simplest way is to reduce value of max_number_matches in parameters file.
- In the estimated poses text file, each pose will be written as a single line representing a row-major 3x4 matrix.
- The last argument mentioned in the command above is optional. When provided, it is used solely to compute translation and rotation error metrics between the estimated and ground truth poses.

## Differences with SOFT2
| SOFT2 | Ours |
| ---                              | ---        |
| Automatic computing of ground plane normal vector and distance to ground plane | The user needs to provide the ground plane normal vector and distance to ground plane if he would like to use patch correction feature |
| Use SOFT algorithm to estimate initial relative pose | Use Lucas–Kanade optical flow to estimate initial matches, followed by a RANSAC to estimate initial relative pose |
| Bundle adjustment | No bundle adjustment
| Extrinsic camera rotation is optimized during scale computing | For now, the extrinsic camera rotation is deemed satisfactory and will not undergo optimization during the algorithm
