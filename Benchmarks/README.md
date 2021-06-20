## Twitch Event

for the members of Twitch Partner- ship Program, the streamer uploads the streaming with the highest bitrate, a.k.a., source quality. The server transcodes the segments into multiple bitrate versions. In our experiment, the streamer uploads R1 segments and the edge server transcodes them into R_2, ..., R_{|R|}.

## Twitch Starter

for the new streamers, the default encoder of Twitch provides fixed bitrate encoding and deliver the streaming to the followers without transcod- ing. In our experiment, the streamer uploads R2 segments without any transcoding at the edge server.

## SAC with decomposition (SAC-w)

the provider uses Soft Actor-Critic framework to estimate the state function and find the inter-frame policy that maximizes the defined reward. Similar to SACCT, the intra-frame actions in SAC-w are determined by the proposed convex optimizer. 

## SAC without decomposition (SAC-wo)

the provider uses Soft Actor-Critic framework to estimate the state function and find both the inter-frame and intra-frame action policies that maximize the defined reward.
