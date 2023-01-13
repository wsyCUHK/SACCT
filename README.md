# SACCT

## Introduction
This is the implementations related to our paper 

S. Wang, S. Bi and Y. -J. A. Zhang, "Deep Reinforcement Learning With Communication Transformer for Adaptive Live Streaming in Wireless Edge Networks," in IEEE Journal on Selected Areas in Communications, vol. 40, no. 1, pp. 308-322, Jan. 2022, doi: 10.1109/JSAC.2021.3126062.

Our paper is highlighed in the recent <a href="https://www.comsoc.org/publications/blogs/selected-ideas-communications/introduction-blog-selected-ideas-communications">JSAC blog</a> and <a href="https://apb.regions.comsoc.org/files/2021/12/AP-Newsletter-No-60-Dec-2021_final_ver.pdf">IEEE Comsoc Asia Pacific Region Newsletter</a>.

The Deep Reinforcement Learning with Communication Transformer model is  upgraded in our recent work. You may further refer to <a href="https://github.com/wsyCUHK/DBAG">Edge Video Analytics with Adaptive Information Gathering: A Deep Reinforcement Learning Approach</a> in TWC.
## Notation Remark
There are some typos in the notation table of the published paper in Page 3. If there is any conflicts, please refer the notations to the followings:

$\mathcal{I}_t$: The set of followers at the beginning of time t 

$\mathcal{I}_t^a$: The set of followers that arrive at the network at time t 

$\mathcal{I}_t^d$: The set of followers that departure from the network at time t

$\mathcal{J}_t$: The set of bitrates that provided to the followers at time t

## Requirements and Installation
We recommended the following dependencies.

* Python 3.6
* Torch 1.8.1
* CUDA 11.1


## About Authors
Shuoyao Wang, sywang[AT]szu[DOT]edu[DOT]cn :Shuoyao Wang received the B.Eng. degree (with first class Hons.) and the Ph.D degree in information engineering from The Chinese University of Hong Kong, Hong Kong, in 2013 and 2018, respectively. From 2018 to 2020, he was an senior researcher with the Department of Risk Management, Tencent, Shenzhen, China. Since 2020, he has been with the College of Electronic and Information Engineering, Shenzhen University, Shenzhen, China, where he is currently an Assistant Professor. His research interests include optimization theory, operational research, and machine learning in Multimedia Processing, Smart Grid, and Communications. See more details in the <a href="https://wsycuhk.github.io/">personal webpage</a>.

This is a co-work with Suzhi Bi and Yingjun Angela Zhang.

## Citation Format
If the implementation helps, you might citate the work with the following foramt:

@ARTICLE{9605672,  
author={Wang, Shuoyao and Bi, Suzhi and Zhang, Ying-Jun Angela},  
journal={IEEE Journal on Selected Areas in Communications},   
title={Deep Reinforcement Learning With Communication Transformer for Adaptive Live Streaming in Wireless Edge Networks},   
year={2022},  volume={40},  number={1},  pages={308-322},  doi={10.1109/JSAC.2021.3126062}}
