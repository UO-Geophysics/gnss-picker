import tensorflow as tf
import numpy as np
from scipy import signal

def make_large_unet_drop(fac, sr, ncomps = 3, winsize = 128):
    
    if ncomps == 1:
        input_layer = tf.keras.layers.Input(shape = (winsize, 2)) # 1 channel seismic data
    elif ncomps == 3:
        input_layer = tf.keras.layers.Input(shape = (winsize, 3)) # 3 channel GNSS data
    
    # First block
    level1 = tf.keras.layers.Conv1D(int(fac*32), 21, activation = 'relu', padding = 'same')(input_layer) # N filters, filter size, stride, padding
    network = tf.keras.layers.MaxPooling1D()(level1) # 32
    
    # Second block
    level2 = tf.keras.layers.Conv1D(int(fac*64), 15, activation = 'relu', padding = 'same')(network)
    network = tf.keras.layers.MaxPooling1D()(level2) # 16
    
    # Next block
    level3 = tf.keras.layers.Conv1D(int(fac*128), 11, activation = 'relu', padding = 'same')(network)
    network = tf.keras.layers.MaxPooling1D()(level3) # 8
    
    # Base of network
    network = tf.keras.layers.Flatten()(network)
    base_level = tf.keras.layers.Dense(16, activation = 'relu')(network)
    network = tf.keras.layers.Reshape((16, 1))(base_level)
    
    # Upsample and add skip connections
    network = tf.keras.layers.Conv1D(int(fac*128), 11, activation = 'relu', padding = 'same')(network)
    network = tf.keras.layers.UpSampling1D()(network)
    
    level3 = tf.keras.layers.Concatenate()([network, level3])
    network = tf.keras.layers.Conv1D(int(fac*64), 15, activation = 'relu', padding = 'same')(level3)
    network = tf.keras.layers.UpSampling1D()(network)
    
    level2 = tf.keras.layers.Concatenate()([network, level2])
    network = tf.keras.layers.Conv1D(int(fac*32), 21, activation = 'relu', padding = 'same')(level2)
    network = tf.keras.layers.UpSampling1D()(network)
    
    level1 = tf.keras.layers.Concatenate()([network, level1])
    
    # End of network
    network = tf.keras.layers.Dropout(.2)(level1)
    network = tf.keras.layers.Conv1D(1, 21, activation = 'sigmoid', padding = 'same')(level1) 
    
    output = tf.keras.layers.Flatten()(network)
    
    model = tf.keras.models.Model(input_layer, output)
    opt = tf.keras.optimizers.Adam(lr = 0.0001)
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
    model.summary()
    
    return model

def my_3comp_data_generator(batch_size, x_data, n_data, meta_data, nan_array, sig_inds, noise_inds, sr, std, valid = False, nlen = 256, winsize = 128):
   
    epsilon = 1e-6
    
    while True:
        

        ### ----- Defining the batch information ----- ###
        
        # Begin by randomly selecting a starting index for the earthquake data batch. This is half the length of the full batch size because the standalone noise samples added later makes the full batch length. Then pick a random starting index for the first noise batch (to be added to the FakeQuakes data) and the second noise batch (which will be standalone).

        start_of_data_batch = np.random.choice(len(sig_inds) - batch_size//2) 
        start_of_noise_batch1 = np.random.choice(len(noise_inds) - batch_size//2) 
        start_of_noise_batch2 = np.random.choice(len(noise_inds) - batch_size//2)
        
        # Then select the right range of indices from each dataset.

        datainds = sig_inds[start_of_data_batch:start_of_data_batch + batch_size//2] 
        noiseinds1 = noise_inds[start_of_noise_batch1:start_of_noise_batch1 + batch_size//2] 
        noiseinds2 = noise_inds[start_of_noise_batch2:start_of_noise_batch2 + batch_size//2] 
        
        ### ----- Making the full batches of data and targets ----- ###
        
        # This section works with the three components of the data separately (order N, E, Z). First the FakeQuakes data is combined with the noise from the first noise batch. The standalone noise from the second batch is then concatenated onto each component, which makes the full batch length. 
        
        # The same "transformation" (concatenation of placeholders for samples which are just noise) is done to the metadata so that the earthquake info is aligned with earthquake samples, and NaNs are aligned with noise samples. 
        
        # The target array is then initialized by concatenating ones (earthquakes) or zeros (standalone noise) in the right shapes. batch_target is defined to make a structure that we can fill with Gaussians when we calculate the actual targets.

        comp1 = np.concatenate((x_data[datainds, :nlen] + n_data[noiseinds1, :nlen], n_data[noiseinds2, :nlen])) # N   
        comp2 = np.concatenate((x_data[datainds, nlen:2*nlen] + n_data[noiseinds1, nlen:2*nlen], n_data[noiseinds2, nlen:2*nlen])) # E
        comp3 = np.concatenate((x_data[datainds, 2*nlen:] + n_data[noiseinds1, 2*nlen:], n_data[noiseinds2, 2*nlen:])) # Z
        metacomp = np.concatenate((meta_data[datainds, :], nan_array[noiseinds2, :]))
        target = np.concatenate((np.ones_like(datainds), np.zeros_like(noiseinds2)))
        batch_target = np.zeros((batch_size, nlen))

        # A new list of indices is created that matches the full batch size. This is randomly shuffled, and then these shuffled indices are used to shuffle the data, metadata, and targets the same way to make sure everything still matches.

        inds = np.arange(batch_size) 
        np.random.shuffle(inds)  
        
        comp1 = comp1[inds, :] 
        comp2 = comp2[inds, :]
        comp3 = comp3[inds, :]
        metacomp = metacomp[inds, :] 
        target = target[inds]
        
        # Now the actual targets are added to batch_target by introducing a Gaussian with a peak aligned with the pick time (currently just in the center of each of the components). A Gaussian will only be added when the target is one, as we set up above, because those are samples with earthquakes.

        for idx, targ in enumerate(target):

            if targ == 0: # Target is zero for this idx = noise sample
                batch_target[idx, :] = np.zeros((1, nlen)) 
                
            elif targ == 1: # Target is one for this idx = earthquake sample
                batch_target[idx, :] = signal.gaussian(nlen, std = int(std * sr))
        
        ### ----- Shifting the pick location so it's not always centered ----- ###
        
        # Next, the samples are shifted so that the pick location is not always centered. This cuts the waveforms down from 256 seconds (257 samples) to 128 seconds (129 samples) or whatever the chosen lengths are. We start by making an array the size of the batch of random time offsets between 0 and 128 seconds, then initalize "new batch" arrays of zeros that we can fill with the shifted data (3D array (batch_size, 128, 3)) and corresponding shifted targets (2D (batch_size, 128) - don't need three components).
        
        # The time offset array is then looped through since it's the same length as the batch. The random offset times are established as the new starting indices for the shortened waveforms, with the end 128 seconds later. The "new batch" of data is then built by taking each component and only selecting out the samples which fall between the new start and end times

        time_offset = np.random.uniform(0, winsize, size = batch_size)

        new_batch = np.zeros((batch_size, int(winsize*sr), 3)) 
        new_batch_target = np.zeros((batch_size, int(winsize*sr)))
            
        for idx, offset in enumerate(time_offset):
            
            offset_sample = int(offset*sr)
            start = offset_sample 
            end = start + int(winsize*sr)

            new_batch[idx, :, 0] = comp1[idx, start:end]
            new_batch[idx, :, 1] = comp2[idx, start:end]
            new_batch[idx, :, 2] = comp3[idx, start:end]
            
            new_batch_target[idx, :] = batch_target[idx, start:end]
    
        # Finally, we get a bunch of outputs depending on the options chosen in the beginning. New batch, as a reminder, is the final shifted data with shuffled noisy earthquakes and standalone noise. We also output the targets (either Gaussians or zeros aligning with the data in new_batch) and the associated metadata.

        if valid: # If valid = True, we are testing and we want the metadata and original data for plotting/analysis
            yield(new_batch, new_batch_target, metacomp)
            
        else: # If valid = False, we are training and only want to give the generator the training data and the targets
            yield(new_batch, new_batch_target)
            
def real_data_generator(data, data_inds, meta_data, sr, std, nlen = 128): # Doesn't use a batch size - just uses the whole thing
   
    epsilon = 1e-6
    
    while True:
        
        ### ----- Making the full batches of data and targets ----- ###
        
        # DATA
        comp1 = data[:, :nlen]
        comp2 = data[:, nlen:2*nlen]
        comp3 = data[:, 2*nlen:]
        
        # TARGETS 
        
        gauss_positions = meta_data[:,5]
        
        simple_target = []
        
        for krow in range(len(gauss_positions)):
            position = gauss_positions[krow]
            if position == 'nan':
                target = 0
            elif position != 'nan':
                target = 1
            simple_target.append(target)
        
        simple_target = np.array(simple_target)  # 1D array that's the length of the data (xxx,) - should be 1s and 0s based on where EQs are by using metadata array times
        
        # Initializing array of zeros to build the array with Gaussians at the pick times
        gauss_target = np.zeros((len(data), nlen)) # 2D array that's length of data x 128. Making structure to hold target functions
        
        ### ----- Making the picks into Gaussians ----- ###
        
        # Adds a Gaussian when the batch_target is one, indicating an earthquake
        for ii, targ in enumerate(simple_target):
        
            if targ == 0: # this is still fine
                gauss_target[ii, :] = np.zeros((1, nlen)) # batch_target was all zeroes before. If the target (whether it's signal or not) is zero, leave batch_target as zero
                # print(batch_target)
                
            elif targ == 1: # need to add to this. Need to use np.roll to shift the Gaussian to line up with the arrival time.
                
                gauss_pos = int(gauss_positions[ii])
                # print(gauss_pos)
                gauss_xs = np.arange(0,128,1)
                
                gauss_ys = []
                
                for kx in range(len(gauss_xs)):
                    
                    gauss_x = gauss_xs[kx]
                    gauss_y = np.exp(-(gauss_x - gauss_pos)**2 / (2 * std**2))
                    
                    gauss_ys.append(gauss_y)
                    
                gauss_ys = np.array(gauss_ys)
                # print(len(gauss_ys))
                
                # plt.plot(gauss_xs,gauss_ys)
            
                gauss_target[ii, :] = gauss_ys # If the target is one, add a Gaussian to the batch to the batch target centered at the pick location

        # Initialize array to hold data
        stack_data = np.zeros((len(data), nlen, 3)) # 3D array - 5000 samples of 3 components, each 128 seconds long
        
        # Stack the components  
        for ii, row in enumerate(stack_data):
            
            # Grabbing out the new batch of data using the shifted timespans
            stack_data[ii, :, 0] = comp1[ii, :] # New N component - row in counter, all 128 columns, 0 for first component. Grabs 128s section from comp1
            stack_data[ii, :, 1] = comp2[ii, :]
            stack_data[ii, :, 2] = comp3[ii, :]
            
        ### ----- Creating batch_out ----- ###
        
        # batch_out = stack_data # The original real data, shape (12240, 128, 3)
        
        yield(stack_data, gauss_target)

def main():
    make_large_unet(1,1,ncomps = 3)

if __name__ == "__main__":
    main()  