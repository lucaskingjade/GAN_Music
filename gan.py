__author__ = 'qiwang'
from keras.models import Sequential
from keras.layers import Dense,Activation,MaxoutDense,InputLayer,Reshape,Flatten
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from load_music import BC4s
import numpy as np

#from keras.utils.visualize_util import plot
from visualize_music import show_samples,show_sample_pairs,plot_objectives

#DATA_DIM = 128
#Noise_Dim = 100
class GAN(object):
    def __init__(self,
                 DATA_DIM,
                 Noise_Dim,
                 BATCH_SIZE,
                 Max_Epochs = 100,):
        args = locals().copy()
        del args['self']
        self.__dict__.update(args)

    #define the Generator
    def Generator(self):
        model = Sequential()
        model.add(Dense(input_dim= self.Noise_Dim,output_dim=1200))
        model.add(Activation('relu'))
        model.add(Dense(output_dim=1200))
        model.add(Activation('relu'))
        model.add(Dense(output_dim=128,activation='sigmoid'))

        return model

    #define the Discriminator
    def Discriminator(self):
        model = Sequential()
        model.add(InputLayer(batch_input_shape=(None,128)))
        model.add(Reshape(target_shape=(1,4,32)))
        model.add(MaxPooling2D(pool_size=(4,4),strides=(2,2)))
        #model.add(MaxoutDense(input_dim=128,output_dim=240,nb_feature=5))
        #model.add(Activation('tanh'))
        #model.add(Dense(output_dim=500))
        #model.add(Activation('tanh'))
        model.add(Flatten())
        model.add(MaxoutDense(input_dim=128,output_dim=240,nb_feature=5))
        model.add(Dense(output_dim=1))
        model.add(Activation('sigmoid'))
        return model

    #define a model for training generator
    def Generator_containing_Discriminator(self,generator,discriminator):
        model = Sequential()
        model.add(generator)
        discriminator.trainable = False
        model.add(discriminator)
        return model

    #def a function for compute the objective values of d and g on different dataset
    def get_loss_g_and_d( self, d,g,g_with_d, \
                          dataset):
        #num_batch = dataset.shape[0]/BATCH_SIZE
        loss_d = 0.0
        loss_g = 0.0
        #for i_batch in range(num_batch):
        #start = i_batch * BATCH_SIZE
        #end = start + BATCH_SIZE
        input_noise = np.random.uniform(-1.0,1.0,(dataset.shape[0], self.Noise_Dim))
        samples = g.predict(input_noise)
        X = np.concatenate((dataset,samples),axis=0)
        Y = [1]*dataset.shape[0] + [0]*dataset.shape[0]
        loss_d = loss_d + d.test_on_batch(X,Y)
        loss_g = loss_g + g_with_d.test_on_batch(input_noise, [1]*dataset.shape[0])
        return loss_d,loss_g



    #def training process
    def train(self,data_obj):
        X_train = data_obj.X_train
        y_train = data_obj.y_train
        X_valid = data_obj.X_val
        y_valid = data_obj.y_val
        X_test  = data_obj.X_test
        y_test  = data_obj.y_test

        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
        X_valid = X_valid.reshape((X_valid.shape[0],X_valid.shape[1]*X_valid.shape[2]))
        discriminator = self.Discriminator()
        generator = self.Generator()
        generator_with_discriminator = self.Generator_containing_Discriminator(generator,discriminator)
        #plot(generator_with_discriminator, to_file='generator_with_discriminator.png')
        d_optim = SGD(lr=0.01,decay = 1e-4,momentum=0.5,nesterov= True)
        g_optim = SGD(lr=0.01,momentum=0.9,nesterov=True)

        generator.compile(optimizer='SGD',loss='binary_crossentropy',metrics=['accuracy'])
        discriminator.compile(loss='binary_crossentropy',optimizer=d_optim)
        generator_with_discriminator.compile(loss='binary_crossentropy',optimizer=g_optim)
        discriminator.trainable=True


        #how to set a seed for random number generator?
        num_batch = int(X_train.shape[0]/self.BATCH_SIZE)
        train_d_loss_his = []
        train_g_loss_his = []
        valid_d_loss_his = []
        valid_g_loss_his = []
        loss_history = dict()
        for epoch in range(self.Max_Epochs):
            print('Epoch seen: {}'.format(epoch))
            print('Number of batches is',num_batch)
            d_loss = 0.0
            g_loss = 0.0
            train_d_loss,train_g_loss = self.get_loss_g_and_d(discriminator,generator,generator_with_discriminator,X_train)
            valid_d_loss,valid_g_loss = self.get_loss_g_and_d(discriminator,generator,generator_with_discriminator,X_valid)
            train_d_loss_his.append(train_d_loss)
            train_g_loss_his.append(train_g_loss)
            valid_d_loss_his.append(valid_d_loss)
            valid_g_loss_his.append(valid_g_loss)
            loss_history['train_d_loss'] = train_d_loss_his
            loss_history['train_g_loss'] = train_g_loss_his
            loss_history['valid_d_loss'] = valid_d_loss_his
            loss_history['valid_g_loss'] = valid_g_loss_his
            plot_objectives(loss_history,'objective.pdf')
            print('Epoch %d, train_d_loss %f'%(epoch,train_d_loss))
            print('Epoch %d, train_g_loss %f'%(epoch,train_g_loss))
            print('Epoch %d, valid_d_loss %f'%(epoch,valid_d_loss))
            print('Epoch %d, valid_g_loss %f'%(epoch,valid_g_loss))
            if abs(valid_d_loss-valid_g_loss)<0.005 and epoch>0:
                print ('generated samples is {}'.format(generated_batch[2]))
            nm_iterations = d_optim.iterations.get_value()
            cur_lr = d_optim.lr.get_value()*(1./(1+d_optim.decay.get_value() * nm_iterations))

            print('Epoch %d, iterations %d, learning_rate %f')%(epoch, nm_iterations, cur_lr)
            for index_batch in range(num_batch):
                #preparing batch
                input_noise = np.random.uniform(-1.0,1.0,(self.BATCH_SIZE, self.Noise_Dim))
                begin=index_batch * self.BATCH_SIZE
                end = begin + self.BATCH_SIZE
                train_batch = X_train[begin:end,:]
                generated_batch = generator.predict(input_noise)

                #train discriminator
                discriminator.trainable = True
                X = np.concatenate((train_batch,generated_batch),axis=0)
                Y = [1] * self.BATCH_SIZE + [0] * self.BATCH_SIZE
                discriminator.train_on_batch(X,Y)
                #d_loss = d_loss + discriminator.train_on_batch(X,Y)
                #a = discriminator.test_on_batch(X,Y)
                #b = generator_with_discriminator.test_on_batch(input_noise,[1]*BATCH_SIZE)
                #print('train_objective_on_d : %f')%(a)
                #print('train_objective_on_g : %f')%(b)
                #print('batch %d, d_loss %f'%(index_batch,discriminator.train_on_batch(X,Y)))

                #train generator
                input_noise = np.random.uniform(-1.0,1.0,(self.BATCH_SIZE, self.Noise_Dim))
                discriminator.trainable = False
                Y_g = [1]*self.BATCH_SIZE
                generator_with_discriminator.train_on_batch(input_noise,Y_g)
                #g_loss = g_loss + generator_with_discriminator.train_on_batch(input_noise,Y_g)
                #generated_batch = generator.predict(input_noise)
                #X = np.concatenate((train_batch,generated_batch),axis=0)
                #Y = [1]*BATCH_SIZE + [0] * BATCH_SIZE
                #a = discriminator.test_on_batch(X,Y)
                #b = generator_with_discriminator.test_on_batch(input_noise,[1]*BATCH_SIZE)
                #print('train_objective_on_d : %f')%(a)
                #print('train_objective_on_g : %f')%(b)
                #print('batch %d, g_loss %f'%(index_batch,generator_with_discriminator.train_on_batch(input_noise,Y_g)))

            #print('Epoch %d, d_loss %f'%(epoch,d_loss/num_batch))
            #print('Epoch %d, g_loss %f'%(epoch,g_loss/num_batch))
            filename1 = 'generated_samples.png'
            filename2 = 'generated_sample_pairs.png'
            show_samples(generator,100,data_obj,filename1)
            show_sample_pairs(generator,100,data_obj,filename2)
            print('Saving model to yaml...')
            generator_yaml = generator.to_yaml()
            with open("generator.yaml","w") as yaml_file:
                yaml_file.write(generator_yaml)

            discriminator_yaml = discriminator.to_yaml()
            with open("discriminator.yaml","w") as yaml_file:
                yaml_file.write(discriminator_yaml)

            generator_with_discriminator_yaml = generator_with_discriminator.to_yaml()
            with open("generator_with_discriminator.yaml",'w') as yaml_file:
                yaml_file.write(generator_with_discriminator_yaml)

            print('Saving weights... ...')
            generator.save_weights('generator.hdf5',overwrite=True)
            discriminator.save_weights('discriminator.hdf5',overwrite=True)
            generator_with_discriminator.save_weights('discriminator.hdf5',overwrite=True)





if __name__ == "__main__":
    gan = GAN(DATA_DIM=128,Noise_Dim=100,BATCH_SIZE=100,Max_Epochs=100)
    data = BC4s(0.6,0.2,patch_len=32,patch_step=4,\
                                   pitch_scale=True,is_one_hot_code=False)
    gan.train(data)





