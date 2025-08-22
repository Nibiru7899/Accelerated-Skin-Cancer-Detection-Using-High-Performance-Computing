# Accelerated Skin Cancer Detection Using High Performance Computing
### Problem Statement:

The goal is to accurately segment skin cancer lesions from a large dataset of dermoscopic images. A naive approach, processing each image sequentially on a single CPU core, is computationally infeasible for the thousands of images required to train a state-of-the-art deep learning model.

This project aims to implement a high-performance pipeline that leverages a hybrid parallel computing model. The pipeline will first be developed as a sequential, single-threaded application on the CPU. It will then be accelerated significantly by parallelizing the data preprocessing stage across multiple CPU cores using **OpenMP** and offloading the computationally expensive model training to a GPU using **CUDA**. The final deliverable will be a comparative analysis of the performance (throughput and training time) achieved by the parallel version over the sequential CPU implementation, along with a validated, high-accuracy segmentation model.

### Step-by-Step Procedure

Here’s a detailed plan broken down into three phases, which can be spread over 2-3 months.

#### Phase 1: Sequential CPU Implementation (Weeks 1-3)

The first step is to build a working, end-to-end pipeline on the CPU. This ensures a correct and functional baseline before introducing the complexities of parallelization.

1.  **Understand the Algorithm:**
    * **Domain Knowledge:** Review the fundamentals of medical image segmentation. The goal is to perform pixel-wise classification on dermoscopic images, assigning each pixel to either "lesion" or "background" to create a binary mask.
    * **Model Architecture (U-Net):** Study how the U-Net architecture works.
        * It consists of an encoder (down-sampling path) to capture context and a decoder (up-sampling path) to enable precise localization.
        * Crucially, it uses "skip connections" to merge feature maps from the encoder to the decoder, which helps recover fine-grained details lost during down-sampling. This makes it exceptionally effective for medical segmentation tasks.

2.  **Setup Development Environment:**
    * Install a C++ compiler (like GCC/G++) for the preprocessing module.
    * Install Python and key libraries: PyTorch, OpenCV, and Scikit-learn.
    * Install the NVIDIA CUDA Toolkit and cuDNN library.

3.  **Implement the Sequential Version (Python & C++):**
    * **Data Structures:** Implement a custom PyTorch `Dataset` class to load images and their corresponding segmentation masks from the ISIC dataset.
    * **Preprocessing:** Write a sequential C++ or Python function that performs all required steps on a single image: artifact removal (hair), color normalization, and resizing.
    * **Model Implementation:** Implement the U-Net model architecture in PyTorch.
    * **Training Loop:** Create a main Python script that, for each training epoch, iterates through the dataset batch by batch, performs the forward pass, calculates the loss (e.g., Dice Loss), runs the backward pass, and updates the model weights.
    * **Verification:** Train the sequential model on a small subset of the data (e.g., 500 images) for 5-10 epochs. Verify that the training loss is decreasing and save a few output masks to visually confirm that the model is learning to identify lesion boundaries.

#### Phase 2: Hybrid Parallelization (Weeks 4-7)

Now, you'll accelerate the two main bottlenecks in the pipeline using OpenMP and CUDA.

1.  **Identify the Bottlenecks:**
    * **Data Preprocessing:** This stage is CPU-bound and I/O-bound. Processing thousands of images one by one takes a significant amount of time.
    * **Model Training:** The forward and backward passes of the U-Net involve billions of floating-point operations (convolutions and matrix multiplications), making it the primary compute-bound bottleneck.

2.  **Parallelize Preprocessing with OpenMP:**
    * The core idea is to process multiple images concurrently on the CPU.
    * Modify your C++ preprocessing module to use the `#pragma omp parallel for` directive on the main loop that iterates over the image files.
    * Alternatively, in the PyTorch `DataLoader`, set `num_workers > 1`. This uses Python's multiprocessing to load and preprocess batches in parallel in the background, ensuring the GPU is not kept waiting for data.

3.  **Accelerate Training with CUDA:**
    * **Data Transfer:** Ensure your main training loop correctly moves data tensors to the GPU using `.to('cuda')`.
    * **Model Execution:** PyTorch will automatically use the highly optimized CUDA and cuDNN libraries to execute the U-Net's operations on the thousands of GPU cores. This requires no change to the model architecture itself.

4.  **Integrate the Parallel Pipeline:**
    * Modify your main training script to use the parallelized data loader (OpenMP/multiprocessing) to feed data efficiently to the U-Net model running on the GPU (CUDA). This creates the complete, high-performance hybrid pipeline.

#### Phase 3: Performance Analysis & Optimization (Weeks 8-10)

This is where you quantify the benefits of the parallel implementation.

1.  **Measure Performance:**
    * Use a simple timer to measure the total execution time for preprocessing a large number of images in the sequential vs. the OpenMP version.
    * Use PyTorch's profiler or CUDA events (`torch.cuda.Event`) to accurately measure the training time per epoch on the GPU.

2.  **Run Experiments:**
    * Execute both the fully sequential CPU pipeline and the accelerated hybrid pipeline on an increasing subset of the dataset (e.g., 1,000, 5,000, and 10,000+ images).
    * For each size, record the total pipeline execution time.

3.  **Analyze and Visualize Results:**
    * Calculate the speedup, defined as `Speedup = Time_sequential / Time_parallel`.
    * Create plots:
        * Preprocessing Time vs. Number of CPU Cores.
        * Total Training Time vs. Dataset Size (for both CPU and GPU versions).
        * Speedup vs. Dataset Size.
        * Model performance (Dice Score) on a validation set vs. Training Epochs.

4.  **(Optional) Optimization & Scaling:**
    * Investigate using Automatic Mixed Precision (AMP) in PyTorch to further speed up training with minimal loss in accuracy.
    * For scaling beyond a single GPU, integrate **Horovod**. This would involve wrapping the PyTorch optimizer and adding a few lines of code to enable distributed training across multiple nodes of the HPC cluster, managed by MPI.
