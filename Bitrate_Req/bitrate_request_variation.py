def user_transit(users,state):
    new_users=[]
    for i in range(users.shape[0]):
        if random.random()>d_rate[users[i,1]]:
            #the user doesnot departure
            new_u=users[i,:]
            if random.random()>d_rate[users[i,1]]:
                #the channel changes
                #if state[2+new_u[0]]>0.5 or (state[2+min(new_u[0]+1,num_of_bitrate-1)]>0.5 and state[2+max(new_u[0]-1,0)]>0.5) or (state[2+min(new_u[0]+1,num_of_bitrate-1)]<0.5 and state[2+max(new_u[0]-1,0)]<0.5):
                if random.random()>0.5:
                    new_u[2]=min(new_u[2]+1,num_of_channel-1)
                else:
                    new_u[2]=max(new_u[2]-1,0)
                score=np.zeros(num_of_bitrate,)
                for j in range(num_of_bitrate):
                    score[j]=user_qoe[0]*candidate_bitrate[j]-(user_qoe[1]*candidate_bitrate[j]/(new_u[2]+1)+user_qoe[2]*abs(candidate_bitrate[j]-candidate_bitrate[new_u[0]]))
                #try:
                #    new_u[0]=np.argmax(score)[0]
                #except:
                new_u[0]=np.argmax(score)

            new_users.append(new_u)
            #New Arrival
    for i in range(3):
        if random.random()<a_rate[i]:
            new_users.append([random.randint(0,num_of_bitrate-1),i,random.randint(0,num_of_channel)])
    return np.array(new_users)