
import tensorflow as tf ### Version (pip install tensorflow-gpu==1.15) (python2)
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn import preprocessing
from numpy import linalg as LA
from pandas.api.types import CategoricalDtype
### Uncomment the sections according to the dataset, fairness constraint (train and test) used, ######

#### The following code will run for compass dataset with Equalized odds as the fairness constraint. ######

######### Adult (Protected = 9) ##############################
#.01 .02 .03 .05 .1 .2 .3, .5
l_val=.1
l_val2=.1
df = pd.read_csv('adult_final.csv', header= None)

mapping = {'<=50k': 0, '>50k': 1}


df = df.replace({14: mapping})


item = list(df[1].unique())
cat_type = CategoricalDtype(categories=item, ordered=True)
df[1] = df[1].astype(cat_type).cat.codes

item = list(df[6].unique())
cat_type = CategoricalDtype(categories=item, ordered=True)
df[6] = df[6].astype(cat_type).cat.codes

item = list(df[13].unique())
cat_type = CategoricalDtype(categories=item, ordered=True)
df[13] = df[13].astype(cat_type).cat.codes

df[3] = df[3].astype(CategoricalDtype(categories=df[3].unique(), ordered=True)).cat.codes
df[5] = df[5].astype(CategoricalDtype(categories=df[5].unique(), ordered=True)).cat.codes
df[7] = df[7].astype(CategoricalDtype(categories=df[7].unique(), ordered=True)).cat.codes
df[8] = df[8].astype(CategoricalDtype(categories=df[8].unique(), ordered=True)).cat.codes
df[9] = df[9].astype(CategoricalDtype(categories=df[9].unique(), ordered=True)).cat.codes


df=df.drop(columns=[4])

print(df.shape[0],df.shape[1])
print(df.head())


#d=df.iloc[0:1158,:]
data=df.values
#data = np.append(df.values, df_test.values, 0)
#data= np.append(data1, d.values, 0)
print(len(data))
l = np.zeros((len(data), 2))
l[np.arange(len(l)), data[:,-1].astype('int')] = 1
data = np.append(data[:,:-1], l, axis=1)
data = data[:48842]




########### BANK (Protected = 0) ##############################
# df = pd.read_csv('bank_train_new.csv')
# data = df.values
# l = np.zeros((len(data), 2))
# l[np.arange(len(l)), data[:,-1].astype('int')] = 1
# data = np.append(data[:,:-1], l, axis=1)
# data = data[:40000]

######### Compas (Protected = 4) #############################
"""
df = pd.read_csv('compass.csv')
data = df.values
l = np.zeros((len(data), 2))
l[np.arange(len(l)), data[:,-1].astype('int')] = 1
data = np.append(data[:,:-1], l, axis=1)
data = data[:5000]

"""
######### German (Protected = 8) #############################
# df = pd.read_csv('german.csv')
# data = df.values
# l = np.zeros((len(data), 2))
# l[np.arange(len(l)), data[:,-1].astype('int')-1] = 1
# data = np.append(data[:,:-1], l, axis=1)



#############################################
print(data.shape)


################### Neurel Network Layers  ####################

#### Layer 1 ###
def dense(inp, inp_shape, hidden_size, soft = 0, name ='dense'):
	with tf.variable_scope(name):
		weights = tf.get_variable("weights", [1, inp_shape, hidden_size], 'float32',initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
		bias = tf.compat.v1.get_variable("bias", [hidden_size], 'float32', initializer=tf.constant_initializer(0, dtype=tf.float32))
		weights = tf.tile(weights, (batch_size, 1, 1))
		out = tf.matmul(inp, weights) + bias
		if soft == 0:
			return tf.nn.relu(out)
		else:
			return out, tf.nn.softmax(temp*out)
#################




sess =  tf.compat.v1.Session()

######### Hyperparameters #################
rho = 10
epsilon = 1e-10
num_epochs =1000
batch_size = 1
learning_rate = 0.001
p = 90.0 ## change for different values of p (for DI)
input_size = (500, data.shape[1]-2) 
hidden_size1 = 500
hidden_size2 = 100
num_classes = 2
temp = 5
prot1 = 8  ## Change the prot based on the protected of the corresponding datatset
prot2= 7
############################################
min_max_scaler = preprocessing.MinMaxScaler()
data[:,:input_size[1]] = min_max_scaler.fit_transform(data[:,:input_size[1]])




############## Build Model #####################
input_data = tf.placeholder(tf.float32, shape=(batch_size, input_size[0], input_size[1]), name="data")
#######
#result = tf.placeholder(tf.float32, shape=(batch_size, input_size[0], 1), name="sens")
########
labels = tf.placeholder(tf.float32, shape=(batch_size, input_size[0], 2), name = "labels")
#prox_inp = tf.concat([input_data[:,:,:9], input_data[:,:,10:]], axis = 2)
fc1_act = dense(input_data, input_size[1], hidden_size1)
fc2_act = dense(fc1_act, hidden_size1, hidden_size2, name = 'dense1')
logits, rounded = dense(fc2_act, hidden_size2, num_classes, soft = 1, name = 'dense2')
lag =  tf.compat.v1.get_variable("lag", (), 'float32', initializer=tf.constant_initializer(rho, dtype=tf.float32))

t_vars =  tf.compat.v1.trainable_variables()[:-1]

#loss_1 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=labels, pos_weight=4.1))
loss_1 = tf.compat.v1.reduce_mean( tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

############ (Uncomment the followinf for traning with) Disparate Impact ####################################
# n_r = tf.reduce_sum(tf.multiply(rounded[:,:,1], 1- input_data[:,:,prot]))
# d_r = tf.reduce_sum(tf.multiply(rounded[:,:,1], input_data[:,:,prot]))
# n_d = n_r / (d_r + epsilon)
# n_d_ = d_r / (n_r + epsilon)
# const = tf.reduce_min(tf.minimum(n_d, n_d_)) - p/100.0
# loss_2 = tf.maximum(-const,0)
###################################


############ (Uncomment the following for traning with) Demographic parity ####################################
#c0 = tf.reduce_sum(tf.multiply(rounded[:,:,1], 1- input_data[:,:,prot]))/ (batch_size*input_size[0])
#c1 = tf.reduce_sum(tf.multiply(rounded[:,:,1], input_data[:,:,prot]))/(batch_size*input_size[0])
#const1 = c0 - c1 
#loss_2 = tf.maximum(const1 - 0.010, 0.0) + tf.maximum(-const1 - 0.010, 0.0)
#loss_2 = tf.abs(const1) - 0.05

############ (Uncomment the following for traning with) Demographic Parity Modified ####################################
'''
#c =  tf.compat.v1.reduce_sum(rounded[:,:,1]) /(batch_size*input_size[0])
c0 =  tf.compat.v1.reduce_sum(tf.multiply(rounded[:,:,1], 1- input_data[:,:,prot1]))/ (tf.reduce_sum(1- input_data[:,:,prot1]))
c1 =  tf.compat.v1.reduce_sum(tf.multiply(rounded[:,:,1], input_data[:,:,prot1]))/(tf.reduce_sum(input_data[:,:,prot1]))
const1 = c0 - c1 
'''





#print(const1)
#c =  tf.compat.v1.add_m(rounded[:,:,1]) /(batch_size*input_size[0])
'''
result=[]
one=tf.ones([1,500],tf.float32)
zero=tf.zeros([1,500],tf.float32)
for i in range(500):
        results = tf.cond( tf.reduce_all(tf.equal(input_data[:,i,prot], 0)),lambda: tf.reduce_sum(tf.add(input_data[:,i,prot], 1)),
        lambda: tf.reduce_sum(tf.multiply(input_data[:,i,prot], 0)))
        result.append(results)

#print(result)

c0 =  tf.compat.v1.add_n(result) / (tf.reduce_sum(result))
c1 =  tf.compat.v1.add_n(1-result) / (tf.reduce_sum(1-result))

c0 =  tf.compat.v1.reduce_sum(tf.multiply(rounded[:,:,1], one-result)) / (tf.reduce_sum(one- result))
c1 =  tf.compat.v1.reduce_sum(tf.multiply(rounded[:,:,1], result)) / (tf.reduce_sum(result))
const1 = c0 - c1 



#print(c0)

c0 =  tf.compat.v1.add_n(tf.where(input_data[:,i,prot1]==0,tf.multiply(rounded[:,i,1], 1- input_data[:,i,prot1]),0) for i in range(500))/ (tf.reduce_sum(1- input_data[:,:,prot1]))
c1 =  tf.compat.v1.add_n(tf.where(input_data[:,:,prot1]==1,tf.multiply(rounded[:,:,1], input_data[:,:,prot1]),0))/(tf.reduce_sum(input_data[:,:,prot1]))
const1 = c0 - c1 
print(const1)

#print(tf.where(y > 0, tf.sqrt(tf.where(y > 0, y, 1)), y)


'''








#################multiple sens

#for gender binary
one=tf.ones([1,500],tf.float32)
zeroes=tf.zeros([1,500],tf.float32)

t0 = (input_data[:,:,prot1] +1)/1
mask0=tf.math.equal(t0, one)
tt0=tf.where(mask0,one,zeroes)

c0 =  tf.compat.v1.reduce_sum(tf.multiply(rounded[:,:,1], tt0))/ (tf.reduce_sum(tt0))
c1 =  tf.compat.v1.reduce_sum(tf.multiply(rounded[:,:,1], 1-tt0))/(tf.reduce_sum(1-tt0))

const00 = c0 - c1 


#for race multiple groups 
t1 = (input_data[:,:,prot2] +1)/1
mask1=tf.math.equal(t1, one)
tt1=tf.where(mask1,one,zeroes)
c2_1 =  tf.compat.v1.reduce_sum(tf.multiply(rounded[:,:,1], tt1))/ (tf.reduce_sum(tt1))
c2_n1 =  tf.compat.v1.reduce_sum(tf.multiply(rounded[:,:,1], 1-tt1))/(tf.reduce_sum(1-tt1))
const1 = c2_1 - c2_n1

t2 = (input_data[:,:,prot2] +1)/2
mask2=tf.math.equal(t2, one)
tt2=tf.where(mask2,one,zeroes)
c2_2 =  tf.compat.v1.reduce_sum(tf.multiply(rounded[:,:,1], tt2))/ (tf.reduce_sum(tt2))
c2_n2 =  tf.compat.v1.reduce_sum(tf.multiply(rounded[:,:,1], 1-tt2))/(tf.reduce_sum(1-tt2))
const2 = c2_2 - c2_n2

t3 = (input_data[:,:,prot2] +1)/3
mask3=tf.math.equal(t3, one)
tt3=tf.where(mask3,one,zeroes)
c2_3 =  tf.compat.v1.reduce_sum(tf.multiply(rounded[:,:,1], tt3))/ (tf.reduce_sum(tt3))
c2_n3 =  tf.compat.v1.reduce_sum(tf.multiply(rounded[:,:,1], 1-tt3))/(tf.reduce_sum(1-tt3))
const3 = c2_3 - c2_n3

t4 = (input_data[:,:,prot2] +1)/4
mask4=tf.math.equal(t4, one)
tt4=tf.where(mask4,one,zeroes)
c2_4 =  tf.compat.v1.reduce_sum(tf.multiply(rounded[:,:,1], tt4))/ (tf.reduce_sum(tt4))
c2_n4 =  tf.compat.v1.reduce_sum(tf.multiply(rounded[:,:,1], 1-tt4))/(tf.reduce_sum(1-tt4))
const4 = c2_4 - c2_n4

t5 = (input_data[:,:,prot2] +1)/5
mask5=tf.math.equal(t5, one)
tt5=tf.where(mask5,one,zeroes)
c2_5 =  tf.compat.v1.reduce_sum(tf.multiply(rounded[:,:,1], tt5))/ (tf.reduce_sum(tt5))
c2_n5 =  tf.compat.v1.reduce_sum(tf.multiply(rounded[:,:,1], 1-tt5))/(tf.reduce_sum(1-tt5))
const5 = c2_5 - c2_n5


const11=const1+const2+const3+const4+const5




loss_2 =  tf.compat.v1.maximum(const00 - l_val, 0.0) +  tf.compat.v1.maximum(-const00 - l_val, 0.0) +  tf.compat.v1.maximum(const11 - l_val2, 0.0) +  tf.compat.v1.maximum(-const11 - l_val2, 0.0)

########### (Uncomment the following for traning with) Equalized Odds ####################################

#c10_0 = tf.reduce_sum(tf.multiply(tf.multiply(rounded[:,:,1], 1- input_data[:,:,prot]), labels[:,:,0]))/ tf.reduce_sum(1- input_data[:,:,prot])
#c10_1 = tf.reduce_sum(tf.multiply(tf.multiply(rounded[:,:,1], input_data[:,:,prot]), labels[:,:,0]))/tf.reduce_sum(input_data[:,:,prot])
#c01_0 = tf.reduce_sum(tf.multiply(tf.multiply(rounded[:,:,0], 1- input_data[:,:,prot]), labels[:,:,1]))/ tf.reduce_sum(1- input_data[:,:,prot])
#c01_1 = tf.reduce_sum(tf.multiply(tf.multiply(rounded[:,:,0], input_data[:,:,prot]), labels[:,:,1]))/tf.reduce_sum(input_data[:,:,prot])
#const1 = tf.abs(c10_0 - c10_1) 
#const2 = tf.abs(c01_0 - c01_1)
#loss_2 = tf.maximum(tf.maximum(const1, const2) - 0.01, 0.0)

# loss_2 = tf.maximum(const1, const2) - 0.04
####################################################################


loss  = loss_1 + (lag * loss_2)  
#loss = loss_1
opt =  tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss, var_list = t_vars)
opt1 =  tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(-loss, var_list = [lag])

# optim = tf.train.AdamOptimizer(learning_rate)
# gvs = optim.compute_gradients(loss, t_vars)
# grad_placeholder = [(tf.placeholder("float", shape=grad[1].get_shape(), name="grad_placeholder"), grad[1]) for grad in gvs[:-1]]
# train_step = optim.apply_gradients(grad_placeholder)

##################################################

mean_acc = []
dp_mean = 0
for cv in range(1):
        '''
        t_ix = np.random.randint(0, 34000, 34000)
        train_data = data[t_ix]
        test_data = data[np.arange(34000, 48500, 14500)]
        '''
        p=np.append(np.arange(0,31654,1),np.arange(0,346,1),axis=0)
        q=np.append(np.arange(31654,45221,1),np.arange(31654,32087,1),axis=0)
        train_data=data[p,:]
        test_data=data[q,:]
        print(len(train_data),len(test_data))
        ################ TRAIN ################################
        tf.compat.v1.global_variables_initializer().run(session=sess)
        saver =  tf.compat.v1.train.Saver(max_to_keep=1)
        #saver.restore(sess, '/Neutron6/manisha.padala/game_theroy/-2')
        start_time = time.time()
        for epoch in range(1,num_epochs):
                log_loss = []
                di_loss = []
                batches = int(len(train_data)/input_size[0])
                for itr in range(batches):
                        data_ = train_data[itr*input_size[0]:(itr+1)*input_size[0], :input_size[1]].reshape(batch_size, input_size[0], input_size[1])
                        true_label = train_data[itr*input_size[0]:(itr+1)*input_size[0], input_size[1]:].reshape(batch_size, input_size[0], num_classes)
                        dict1 = { input_data: data_, labels: true_label}
                        # g = sess.run(gvs[:-1],feed_dict= dict1)
                        # dict2 = { i[0]: d[0] for i, d in zip(grad_placeholder, g)}
                        # dict1.update(dict2)
                        # _,_,_,l1, l2, l = sess.run([train_step, op1, op2, loss_1, loss_2, loss],feed_dict= dict1)
                        #print(w)
                        _, l1, l2, l = sess.run([opt, loss_1, loss_2, loss], feed_dict = dict1)
                        _, lam, l2, l = sess.run([opt1, lag, loss_2, loss], feed_dict = dict1)
                        
                #if epoch%1000 == 0 or epoch == 1:
                #	print(lam)
                        
                        

        ################ TEST #################################
        #print(di_/batches)
        plot_data = np.zeros((len(test_data), 3))
        labels_predicted  = []
        correct = 0
        total = 0
        di_ = 0
        TP=0
        FP=0
        FN=0
        true_label1 = []
        batches = int(len(test_data)/input_size[0])
        for itr in range(batches):
                data_ = test_data[itr*input_size[0]:(itr+1)*input_size[0], :input_size[1]].reshape(batch_size, input_size[0], input_size[1])
                true_label = test_data[itr*input_size[0]:(itr+1)*input_size[0], input_size[1]:].reshape(batch_size, input_size[0], num_classes)
                
                pred_ = sess.run([rounded],feed_dict={ input_data: data_, labels: true_label})
                #print(di, 1.0/(di+epsilon))
                #di_ += 1.0/(di+epsilon)
                pred_ = pred_[0]
                
                true_label1.extend(np.argmax(true_label[0],1).flatten())
                labels_predicted.extend(np.argmax(pred_[0],1).flatten())
                
                plot_data[itr*input_size[0]:(itr+1)*input_size[0], 0] = data_[0,:,4]
                plot_data[itr*input_size[0]:(itr+1)*input_size[0], 1] = np.argmax(true_label[0],1).flatten()
                plot_data[itr*input_size[0]:(itr+1)*input_size[0], 2] = np.argmax(pred_[0],1).flatten()
                correct += list(np.argmax(pred_[0],1).flatten() == np.argmax(true_label[0],1).flatten()).count(True)
                total += len(true_label[0])
                
              
                
                
                
                


        labels_predicted = np.array(labels_predicted)
        # DP ###
        #t1 = np.sum(labels_predicted * test_data[:,prot])/ len(test_data) 
        #t0 = np.sum(labels_predicted * (1 - test_data[:,prot]))/ len(test_data)
        #dp_test = np.maximum(t0 -t1, t1-t0)
        #dp_mean += dp_test
        np.save('plot_'+ str(cv)+'.npy', plot_data)

        #### DP Modified ########
       
        t = np.sum(labels_predicted)/ np.float(len(test_data))
        
        t1 = np.sum(labels_predicted * test_data[:,prot1])/ np.sum(test_data[:, prot1]) 
        t0 = np.sum(labels_predicted * (1 - test_data[:,prot1]))/ np.sum(1 - test_data[:,prot1])
        
        TP=0
        TN=0
        FP=0
        FN=0
        TP_af=0
        TN_af=0
        FP_af=0
        FN_af=0
        
        precision=0
        recall=0
        
        count1=0
        count2=0
        
        accr1=0
        accr2=0
        print(labels_predicted,true_label1)
        for i in range(0,13567):
                if test_data[i,prot1]==1:
                   # print(labels_predicted[i], true_label1[i])
                    if labels_predicted[i]==1 and true_label1[i]==1:
                        TP_af=TP_af+1
                    if labels_predicted[i]==0 and true_label1[i]==0:
                        TN_af=TN_af+1    
                    if labels_predicted[i]==1 and true_label1[i]==0:
                        FP_af=FP_af+1 
                    if labels_predicted[i]==0 and true_label1[i]==1:
                        FN_af=FN_af+1
                    count1=count1+1
                else:
                    #print(labels_predicted[i], true_label1[i])
                    
                    if labels_predicted[i]==1 and true_label1[i]==1:
                        TP=TP+1
                    if labels_predicted[i]==0 and true_label1[i]==0:
                        TN=TN+1
                    if labels_predicted[i]==1 and true_label1[i]==0:
                        FP=FP+1 
                    if labels_predicted[i]==0 and true_label1[i]==1:
                        FN=FN+1
                    count2=count2+1
                            
        print("TP,FP,TN,FN")
        print(TP_af,FP_af,TN_af,FN_af)
        precision=float(TP_af/(TP_af+FP_af))
        recall=float(TP_af/(TP_af+FN_af))
        accr1=float((TP_af+FP_af)/(count1))
        print("for g1 precision", precision)
        print("for g1 recall", recall)
        print("for g1 accept rate", accr1)

        print(TP,FP,TN,FN)
        precision=float(TP/(TP+FP))
        recall=float(TP/(TP+FN))
        accr2=float((TP+FP)/(count2))
        print("for g2 precision", precision)
        print("for g2 recall", recall)
        print("for g2 accept rate", accr2)

       
       
       
       
       
       ########################race#####
       
        
        
        precision=0
        recall=0
        
        TP_r0=0
        TN_r0=0
        FP_r0=0
        FN_r0=0
        TP_r1=0
        TN_r1=0
        FP_r1=0
        FN_r1=0
        TP_r2=0
        TN_r2=0
        FP_r2=0
        FN_r2=0
        TP_r3=0
        TN_r3=0
        FP_r3=0
        FN_r3=0
        TP_r4=0
        TN_r4=0
        FP_r4=0
        FN_r4=0
        
        count1=0
        count2=0
        count3=0
        count4=0
        count5=0
        
        
        accr3=0
        accr4=0
        accr5=0
        accr6=0
        accr7=0
        
        print(labels_predicted,true_label1)
        for i in range(0,13567):
                if test_data[i,prot2]==0:
                   # print(labels_predicted[i], true_label1[i])
                    if labels_predicted[i]==1 and true_label1[i]==1:
                        TP_r0=TP_r0+1
                    if labels_predicted[i]==0 and true_label1[i]==0:
                        TN_r0=TN_r0+1    
                    if labels_predicted[i]==1 and true_label1[i]==0:
                        FP_r0=FP_r0+1 
                    if labels_predicted[i]==0 and true_label1[i]==1:
                        FN_r0=FN_r0+1
                    count1=count1+1
                elif test_data[i,prot2]==1:
                    if labels_predicted[i]==1 and true_label1[i]==1:
                        TP_r1=TP_r1+1
                    if labels_predicted[i]==0 and true_label1[i]==0:
                        TN_r1=TN_r1+1    
                    if labels_predicted[i]==1 and true_label1[i]==0:
                        FP_r1=FP_r1+1 
                    if labels_predicted[i]==0 and true_label1[i]==1:
                        FN_r1=FN_r1+1
                    count2=count2+1
                elif test_data[i,prot2]==2:
                   # print(labels_predicted[i], true_label1[i])
                    if labels_predicted[i]==1 and true_label1[i]==1:
                        TP_r2=TP_r2+1
                    if labels_predicted[i]==0 and true_label1[i]==0:
                        TN_r2=TN_r2+1    
                    if labels_predicted[i]==1 and true_label1[i]==0:
                        FP_r2=FP_r2+1 
                    if labels_predicted[i]==0 and true_label1[i]==1:
                        FN_r2=FN_r2+1
                    count3=count3+1
                elif test_data[i,prot2]==3:
                    #print(labels_predicted[i], true_label1[i])
                    
                    if labels_predicted[i]==1 and true_label1[i]==1:
                        TP_r3=TP_r3+1
                    if labels_predicted[i]==0 and true_label1[i]==0:
                        TN_r3=TN_r3+1    
                    if labels_predicted[i]==1 and true_label1[i]==0:
                        FP_r3=FP_r3+1 
                    if labels_predicted[i]==0 and true_label1[i]==1:
                        FN_r3=FN_r3+1
                    count4=count4+1  
                elif test_data[i,prot2]==4:
                    #print(labels_predicted[i], true_label1[i])
                    
                    if labels_predicted[i]==1 and true_label1[i]==1:
                        TP_r4=TP_r4+1
                    if labels_predicted[i]==0 and true_label1[i]==0:
                        TN_r4=TN_r4+1    
                    if labels_predicted[i]==1 and true_label1[i]==0:
                        FP_r4=FP_r4+1 
                    if labels_predicted[i]==0 and true_label1[i]==1:
                        FN_r4=FN_r4+1
                    count5=count5+1      
                            
        print("TP_0,FP_0,TN_0,FN_0")
        print(TP_r0,FP_r0,TN_r0,FN_r0)
        precision=float(TP_r0/(TP_r0+FP_r0))
        recall=float(TP_r0/(TP_r0+FN_r0))
        accr3=float((TP_r0+FP_r0)/(count1))
        print("for r0 precision", precision)
        print("for r0 recall", recall)
        print("for r0 accept rate", accr3)

        print("TP_1,FP_1,TN_1,FN_1")
        print(TP_r1,FP_r1,TN_r1,FN_r1)
        precision=float(TP_r1/(TP_r1+FP_r1))
        recall=float(TP_r1/(TP_r1+FN_r1))
        accr4=float((TP_r1+FP_r1)/(count2))
        print("for r1 precision", precision)
        print("for r1 recall", recall)
        print("for r1 accept rate", accr4)
        
        print("TP_2,FP_2,TN_0,FN_0")
        print(TP_r2,FP_r2,TN_r2,FN_r2)
        precision=float(TP_r2/(TP_r2+FP_r2))
        recall=float(TP_r2/(TP_r2+FN_r2))
        accr5=float((TP_r2+FP_r2)/(count3))
        print("for r2 precision", precision)
        print("for r2 recall", recall)
        print("for r2 accept rate", accr5)

        print("TP_3,FP_3,TN_3,FN_3")
        print(TP_r3,FP_r3,TN_r3,FN_r3)
        precision=float(TP_r3/(TP_r3+FP_r3))
        recall=float(TP_r3/(TP_r3+FN_r3))
        accr6=float((TP_r3+FP_r3)/(count4))
        print("for r3 precision", precision)
        print("for r3 recall", recall)
        print("for r3 accept rate", accr6)
        
        print("TP_4,FP_4,TN_4,FN_4")
        print(TP_r4,FP_r4,TN_r4,FN_r4)
        precision=float(TP_r4/(TP_r4+FP_r4))
        recall=float(TP_r4/(TP_r4+FN_r4))
        accr7=float((TP_r4+FP_r4)/(count5))
        print("for g1 precision", precision)
        print("for g1 recall", recall)
        print("for g1 accept rate", accr7)
       ##################################
       
       
       
       
       


        TP=0
        FP=0
        TN=0
        FN=0
        precision=0
        recall=0
        acc1=0
        acc2=0
        
        #for i in range(0,len(labels_predicted)):
        for i in range(0,13567):
           # print(labels_predicted[i] ,true_label1[i])
            if labels_predicted[i]==1 and true_label1[i]==1:
                TP=TP+1
            if labels_predicted[i]==0 and true_label1[i]==0:
                TN=TN+1     
            if labels_predicted[i]==1 and true_label1[i]==0:
                FP=FP+1 
            if labels_predicted[i]==0 and true_label1[i]==1:
                FN=FN+1
        precision=float(TP/(TP+FP))
        recall=float(TP/(TP+FN))
        acc1=float((TP+FP)/(TP+FP+TN+FN))
        acc2=float((TP+TN)/(TP+FP+TN+FN))
        print(TP,FP,TN,FN)
        print("for all precision", precision)
        print("for all recall", recall)
        print("for all acc rate", acc1)
        print("for all accuracy:", acc2)
        Dm_pa=accr1-accr2
        print("Demo-Parity:",Dm_pa)       
        
        dp_test =  np.abs(t1-t0)
        print(t1,t0)
       
        dp_mean += dp_test

        ## EO ###
        #t10_1 = np.sum(test_data[:,-2] * labels_predicted * test_data[:,prot])/ np.sum((test_data[:,prot])) 
        #t10_0 = np.sum(test_data[:,-2] * labels_predicted * (1 - test_data[:,prot]))/ np.sum((1 - test_data[:,prot]))
        #t01_1 = np.sum(test_data[:,-1] * (1 - labels_predicted) * test_data[:,prot])/ np.sum((test_data[:,prot])) 
        #t01_0 = np.sum(test_data[:,-1] * (1 - labels_predicted) * (1 - test_data[:,prot]))/ np.sum((1 - test_data[:,prot]))
        #eo_test = np.maximum(np.abs(t10_1 - t10_0), np.abs(t01_1 - t01_0))
        #dp_mean += eo_test

        #### DI ###########
        # nr = np.sum(labels_predicted * test_data[:,prot])
        # dr = np.sum(labels_predicted * (1 - test_data[:,prot]))
        # print('DI value of the prediction:', nr/dr, dr/nr)
        #print('DP of the test', eo_test)
        #dp_mean += np.minimum(nr/dr, dr/nr)

        print(dp_test)
        print('Accuracy of the network: {} %'.format(100.0 * correct / total))
        mean_acc.append((100.0 * correct / total))


print('Mean accuracy of the network, dp',np.mean(mean_acc), dp_mean/5.0)


#################################################################################################



