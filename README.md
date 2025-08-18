# HINNet: Inertial navigation with head-mounted sensors using a neural network

[cite_start]This repository is associated with the paper "HINNet: Inertial navigation with head-mounted sensors using a neural network," published in *Engineering Applications of Artificial Intelligence*[cite: 2].

## Introduction

[cite_start]Human inertial navigation systems are rapidly developing and have shown great potential in healthcare, smart homes, sports, and emergency services[cite: 23]. [cite_start]Placing inertial measurement units (IMUs) on the head for localization is a relatively new approach but offers an interesting option as many everyday head-worn items could be equipped with sensors[cite: 24, 25]. [cite_start]However, there is a lack of research in this area, and no current solutions allow for free head rotations during long periods of walking[cite: 26].

[cite_start]To address this, we present HINNet, the first deep neural network (DNN) pedestrian inertial navigation system that allows for free head movements with head-mounted IMUs[cite: 27]. [cite_start]HINNet deploys a 2-layer bi-directional Long Short-Term Memory (LSTM) network[cite: 27]. [cite_start]A novel 'peak ratio' feature is introduced and used as part of the input to the neural network to differentiate between head movements and changes in walking patterns[cite: 28, 29].

[cite_start]A dataset was collected from 8 subjects, totaling 528 minutes of data on three different tracks for training and verification[cite: 30].

## Dataset Link

[cite_start]The original dataset can be found on GitHub: [HINNet_Dataset](https://github.com/xinyuhou/HINNet) [cite: 15, 399]

### Trajectories
Trajectories of HINNet Method: 
![screenshot](results/img1.png)
![screenshot2](results/img2.png)
![screenshot3](results/img3.png)
