## Bitrate Rate
The follower decides its bitrate according to its QoE function. In particular, we use the QoE function as $w_1r-w_2r/channel_rate-w_3|r_{-1}-r|$
![bitrate_random_walk](https://user-images.githubusercontent.com/37823466/105622374-16eba700-5e4c-11eb-956c-6512bd3ff400.png)
### The training data file contrains 4 matrices：
./data/data118_traintest.mat: There are 4 variables inside our .mat file, i.e., training_x, training_y, testing_x, testing_y.   
        training_x:110000×180  
        training_y:110000×180  
        testing_x:10000×180  
        testing_y:10000×180  

### The training data is generated as follows:  
Step 1: We first generate 110000 sets of base load data by extending the real-world data through artificially generating the loads on each bus with normal distribution.  
Step 2: We randomly select 10000 sets of loads to implement FDIA.  
Step 3: For each attack, we randomly select a set of target state variables to compromise. In particular, the number of target state variables is uniformly distributed between 2-5 and the indices of target state variables are uniformly chosen from all the state variables.  
Step 4: For each set of load and its particular target state variables, we generate stealthy FDIA according to the min-cut algorithm in [Bi & Zhang 2014] ( S. Bi and Y. J. Zhang, “Using covert topological information for defense against malicious attacks on DC state estimation,” IEEE J. on Selected Areas in Comm., vol. 32, no. 7, pp. 1471–1485, July 2014.).  
Step 5: Due to the dynamic measurement noise, we also append random Gaussian noises to both compromised and uncompromised data.  

After we generate the training data, we repeat above process for 10 times to generate 10 independent testing
data, which naturally introduces validation variations. 
