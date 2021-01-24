## Bitrate Rate
The follower decides its bitrate according to its QoE function. In particular, we use the QoE function as w_1r-w_2r/channel_capacity-w_3|r_{-1}-r| in out simulation. The channel capacity follows a Markov Process as the following figure.
![bitrate_random_walk](https://user-images.githubusercontent.com/37823466/105622374-16eba700-5e4c-11eb-956c-6512bd3ff400.png)

The implementation can be found in bitrate_request_variation.py
