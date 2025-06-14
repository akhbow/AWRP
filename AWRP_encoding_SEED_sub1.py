import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io
from pyts.image import RecurrencePlot
import time
import psutil
from multiprocessing import Pool
import os
from PIL import Image
import re
import math

# Trials for subject 1
keys = ['djc_eeg1', 'djc_eeg2', 'djc_eeg3', 'djc_eeg4', 'djc_eeg5', 'djc_eeg6',
        'djc_eeg7', 'djc_eeg8', 'djc_eeg9', 'djc_eeg10', 'djc_eeg11', 'djc_eeg12',
        'djc_eeg13', 'djc_eeg14', 'djc_eeg15']

class Constants:
    EMBEDDING_DIMENSION = 9
    TIME_DELAY = 550
    SEGMENT_LENGTH = 8
    PERCENTAGE_THRESHOLD = 10
    NUM_CHANNEL = 64

# The window size, denoted as SEGMENT_LENGTH, is constrained to be a perfect square. This requirement necessitates adherence to the following formula:
# SEGMENT_LENGTH = 2 * i^2
# where i is a positive integer belonging to the set {1, 2, 3, 4, ...}.

class RecurrencePlotGenerator:
    def __init__(self):
        self.rp = RecurrencePlot(dimension=Constants.EMBEDDING_DIMENSION,
                                 time_delay=Constants.TIME_DELAY,
                                 threshold=None,
                                 percentage=Constants.PERCENTAGE_THRESHOLD)

    def generate_rp(self, signal):
        sig2D = signal.reshape(1, -1)
        return self.rp.fit_transform(sig2D)[0]

class DataProcessor:
    def __init__(self, subjects):
        self.subjects = subjects

    def read_subject(self, i):
        subject = scipy.io.loadmat(self.subjects[i])
        return subject

    def read_data(self, i, trial_key):
        data = self.read_subject(i)[trial_key]
        return data

    def get_signal_segment(self, signal, win):
        # split = round(len(signal) / Constants.SEGMENT_LENGTH)
        split = math.floor(len(signal) / Constants.SEGMENT_LENGTH)
        start_idx = win * split
        end_idx = min((win + 1) * split, len(signal))
        return signal[start_idx:end_idx]
  
class ImageProcessor:
    @staticmethod
    def combine_rp_images(rp1, rp2):
        if rp1.shape != rp2.shape:
            max_shape = max(rp1.shape[0], rp2.shape[0]), max(rp1.shape[1], rp2.shape[1])
            rp1 = cv2.resize(rp1, max_shape, interpolation=cv2.INTER_LINEAR)
            rp2 = cv2.resize(rp2, max_shape, interpolation=cv2.INTER_LINEAR)
        combined = (rp1 + rp2) / 2
        angle = np.pi / 4
        y, x = rp1.shape
        yy, xx = np.mgrid[:y, :x]
        rp1_mask = (xx - x) * np.tan(angle) - 0.5 > (yy - y)
        rp2_mask = (xx - x) * np.tan(angle) + 0.5 < (yy - y)
        combined[rp1_mask] = rp1[rp1_mask]
        combined[rp2_mask] = rp2[rp2_mask]
        return combined

    @staticmethod
    # Construct the AWRP from ARPs in memory
    def construct_awrp_from_arps(imgs, window_size):
        # Check if the total number of images is equal to the window size
        if len(imgs) != window_size/2:
            raise ValueError("The total number of images must be equal to the specified window size.")

        # Determine the number of rows and images per row
        images_per_row = int((window_size/2) ** 0.5)  # Assuming the window forms a square grid

        # Stitch images row-wise and then concatenate rows to form the final AWRP
        awrp_rows = []
        for row in range(images_per_row):
            start_index = row * images_per_row
            end_index = start_index + images_per_row
            awrp_row = np.concatenate(imgs[start_index:end_index], axis=1)
            awrp_rows.append(awrp_row)

        awrp = np.concatenate(awrp_rows, axis=0)
        
        return awrp

    @staticmethod
    def save_awrp_for_trial(sub, trial_key, rp_generator, data_processor):
        for chn in range(Constants.NUM_CHANNEL):
            signal = data_processor.read_data(sub, trial_key)[chn]
            rps = []
            for win in range(Constants.SEGMENT_LENGTH):
                segment = data_processor.get_signal_segment(signal, win)
                if len(segment) > 0:
                    rp = rp_generator.generate_rp(segment)
                    rps.append(rp)
            
            rps = [rp for rp in rps if rp is not None]
            arps = [ImageProcessor.combine_rp_images(rps[i], rps[i + 1]) for i in range(0, len(rps), 2)]
            awrp_img = ImageProcessor.construct_awrp_from_arps(arps, Constants.SEGMENT_LENGTH)

            temp_filename = f'temp_image_sub_01_exp_{sub}_trial_{trial_key}_ch_{chn}.png'

            plt.figure(figsize=(5, 5))
            plt.imshow(awrp_img, cmap="jet")
            plt.axis('off')
            plt.savefig(temp_filename, bbox_inches='tight', pad_inches=0)
            plt.close()

            try:
                if os.path.exists(temp_filename):
                    img = Image.open(temp_filename)
                    img_resized = img.resize((918, 918), Image.ANTIALIAS)

                    # Extract the trial number from the trial_key (assuming 'djc_eeg' is a constant prefix)
                    trial_number = int(re.search(r'djc_eeg(\d+)', trial_key).group(1))

                    filename = f'AWRP_SEED_{Constants.SEGMENT_LENGTH}/sub_01_exp_{sub+1:02d}_trial_{trial_number:02d}_ch_{chn+1:02d}.png'
                    img_resized.save(filename)

                    img.close()
                    os.remove(temp_filename)
            except Exception as e:
                print(f"Error processing file {temp_filename}: {e}")


def generate_and_record_awrp(args):
    sub, trial_key = args
    # Initialize classes
    subjects = ['sub_01_1.mat','sub_01_2.mat','sub_01_3.mat']
    data_processor = DataProcessor(subjects)
    rp_generator = RecurrencePlotGenerator()
    image_processor = ImageProcessor()

    initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    start_time = time.time()

    # Generate AWRP
    image_processor.save_awrp_for_trial(sub, trial_key, rp_generator, data_processor)

    final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    memory_used = final_memory - initial_memory
    time_taken = time.time() - start_time

    return memory_used, time_taken

def main():
        # Create AWRP directory
    try:
        os.makedirs(f'AWRP_SEED_{Constants.SEGMENT_LENGTH}')
    except FileExistsError:
        pass

    with Pool() as pool:
        results = pool.map(generate_and_record_awrp, [(sub, trial_key) for sub in range(3) for trial_key in keys])

    memory_records = [res[0] for res in results]
    time_records = [res[1] for res in results]

    # Open a text file in write mode
    with open('avg_time-memory_usage_AWRP_{Constants.SEGMENT_LENGTH}.txt', 'w') as file:
        file.write(f"Total Memory Used: {sum(memory_records)} MB\n")
        file.write(f"Average Memory Per AWRP: {sum(memory_records)/len(memory_records)} MB\n")
        file.write(f"Standard Deviation of Memory Per AWRP: {np.std(memory_records)} MB\n")

        file.write(f"Total Time Taken: {sum(time_records)} seconds\n")
        file.write(f"Average Time Per AWRP: {sum(time_records)/len(time_records)} seconds\n")
        file.write(f"Standard Deviation of Time Per AWRP: {np.std(time_records)} seconds\n")

if __name__ == '__main__':
    main()