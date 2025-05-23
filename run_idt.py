import os
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA
import pandas as pd
import re

import densetrack

# Choose which descriptors you would like to run. If all are set to False,
# raw IDT features will be outputted.
HOG_FISHER_VECTOR = True
HOF_FISHER_VECTOR = True
MBH_FISHER_VECTOR = True

DATA_DIRECTORY = 'data'  # the directory of the input video files
TARGET_DIRECTORY = 'features'  # targets will be put in this directory

K = 128  # GMM components for the Fisher Vector


def read_video(file):
    """
    Reads the frames from a video file.

    :param file: the filename in the data directory.
    :type file: String

    :return: the gray video as a numpy array.
    :rtype: array_like, shape (D, H, W) where D is the number of frames,
            H and W are the resolution.
    """
    vidcap = cv2.VideoCapture(os.path.join(DATA_DIRECTORY, file))

    video = []
    success, image = vidcap.read()
    while success:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        video.append(gray)
        success, image = vidcap.read()

    return np.array(video)
    
    
def fisher_vector(xx, gmm):
    """
    Computes the fisher vector on a set of descriptors and GMM.

    :param xx: the descriptor (e.g. HOG/HOF/MBH) for each video.
    :type xx: array_like, shape (N, D) where N is the number of descriptors
              and D is the dimension of each descriptor (e.g. 96 for HOG).

    :param gmm: Gauassian mixture model of the descriptors.
    :type gmm: instance of sklearn mixture.GMM object

    :return: Fisher vector of the given descriptor.
    :rtype: array_like, shape (K + 2 * D * K, ) where K is the number of GMM
            components and D is the dimension of each descriptor.

    This function was taken from this GitHub gist: https://gist.github.com/danoneata/9927923
    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covariances_
        + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))


def read_hog(descriptors):
    """
    Extracts the HOG descriptors from the improved dense trajectory features.

    :param descriptors: the improved dense trajectories returned by
                        densetrack.densetrack method.
    :type descriptors: array_like, shape (N, ) where N is the number of
                       trajectories. Value of each element is a np.void.

    :return: HOG descriptors.
    :rtype: array_like, shape (N, ) where N is the number of trajectories.
    """
    return np.array([descriptors[i][-5] for i in
                    range(descriptors.shape[0])])


def read_hof(descriptors):
    """
    Extracts the HOF descriptors from the improved dense trajectory features.

    :param descriptors: the improved dense trajectories returned by
                        densetrack.densetrack method.
    :type descriptors: array_like, shape (N, ) where N is the number of
                       trajectories. Value of each element is a np.void.

    :return: HOF descriptors.
    :rtype: array_like, shape (N, ) where N is the number of trajectories.
    """
    return np.array([descriptors[i][-4] for i in
                     range(descriptors.shape[0])])


def read_mbh(descriptors):
    """
    Extracts the MBH descriptors from the improved dense trajectory features.

    :param descriptors: the improved dense trajectories returned by
                        densetrack.densetrack method.
    :type descriptors: array_like, shape (N, ) where N is the number of
                       trajectories. Value of each element is a np.void.

    :return: MBH descriptors.
    :rtype: array_like, shape (N, ) where N is the number of trajectories.
    """
    mbh_x_descriptors = np.array([descriptors[i][-3] for i in
                                  range(descriptors.shape[0])])
    mbh_y_descriptors = np.array([descriptors[i][-2] for i in
                                  range(descriptors.shape[0])])

    return mbh_x_descriptors, mbh_y_descriptors


def fisher_descriptor(descriptor):
    """
    Compute the fisher vector based on a set of descriptors.

    As described in the original Improved Dense Trajectory papers, the dimension
    of the descriptors is halved with PCA. The default value of the GMM
    components, K (defined at the top of the script), is 256.

    :param descriptor: the descriptor (e.g. HOG/HOF/MBH) for each video.
    :type descriptor: array_like, shape (N, D) where N is the number of
                      descriptors and D is the dimension of each descriptor
                      (e.g. 96 for HOG).

    :return: fisher vector of the descriptors.
    :rtype: array_like, shape (K + 2 * K * D/2) since the dimension is halved
            with a PCA.
    """
    # Ensure descriptor is 2D - reshape if it has more than 2 dimensions
    if descriptor.ndim > 2:
        # Get the size of first descriptor to determine feature dimension
        feature_dim = np.prod(descriptor[0].shape)
        # Reshape to 2D: (n_samples, feature_dimension)
        descriptor = np.array([d.flatten() for d in descriptor])

    descriptor_pca = PCA(n_components=int(descriptor.shape[1]/2))\
        .fit(descriptor).transform(descriptor)
    gmm = GMM(n_components=K, covariance_type='diag').fit(descriptor_pca)
    return fisher_vector(descriptor_pca, gmm)


def add_row_to_table(data_frame, name, features):
    """
    Adds a row to the data frame.
    If empty, creates the data frame. Otherwise, add name of the video and
    corresponding descriptor's features.
    """
    if data_frame.empty:
        col_names = ['name'] + [str(i) for i in range(len(features))]
        dtypes = {'name': 'object'}
        for i in range(len(features)):


            dtypes[str(i)] = 'float64'
        data_frame = pd.DataFrame(columns=col_names).astype(dtypes)
    # Create new row dictionary
    new_row_data = {'name': name}
    for i, val in enumerate(features):
        new_row_data[str(i)] = val

    # Append to the DataFrame
    data_frame = pd.concat([data_frame, pd.DataFrame([new_row_data], columns=data_frame.columns)], ignore_index=True)

    return data_frame



def save_as_csv(data_frame, path):
    """
    Creates a CSV file from a data frame and does nothing if frame is empty.

    :param data_frame: The final data frame of a descriptor (HOG/HOG/MBH) for
                       all videos.
    :type data_frame: DataFrame

    :param path: The path to write the DataFrame to.
    :type path: String
    """
    if not data_frame.empty:
        data_frame.to_csv(path)

def sorted_alphanumeric(data):
    """
    Sorts alphanumerically.
    :param data: data to be sorted e.g. a list
    :return: sorted data alphanumerically e.g. "User_2" before "User_10"
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def main():
    if not os.listdir(DATA_DIRECTORY):
        print('No input video files are present. Please put your files in the '
              '"data" directory, rebuild the image and run the container.')
    else:
        hog_df, hof_df, mbh_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        for i, file in enumerate(sorted_alphanumeric(os.listdir(DATA_DIRECTORY))):
            video = read_video(file)
            print('------------------------------------------')
            print(f'Running: {file} of shape {video.shape}')

            tracks = densetrack.densetrack(video, adjust_camera=True)
            name = os.path.splitext(os.path.split(file)[1])[0]

            if not (HOG_FISHER_VECTOR or HOF_FISHER_VECTOR or
                    MBH_FISHER_VECTOR):
                # save all trajectories and descriptors without processing
                np.save(os.path.join(TARGET_DIRECTORY, name
                                     + '-trajectory'), tracks)
            else:
                if HOG_FISHER_VECTOR:
                    hog_descriptors = read_hog(tracks)
                    hog_fv = fisher_descriptor(hog_descriptors)
                    hog_df = add_row_to_table(hog_df, name, hog_fv)
                if HOF_FISHER_VECTOR:
                    hof_descriptors = read_hof(tracks)
                    hof_fv = fisher_descriptor(hof_descriptors)
                    hof_df = add_row_to_table(hof_df, name, hof_fv)

                if MBH_FISHER_VECTOR:
                    mbh_x_descriptors, mbh_y_descriptors = read_mbh(tracks)
                    mbh_x_fv = fisher_descriptor(mbh_x_descriptors)
                    mbh_y_fv = fisher_descriptor(mbh_y_descriptors)

                    mbh_fv = np.concatenate((mbh_x_fv, mbh_y_fv))
                    mbh_df = add_row_to_table(mbh_df, name, mbh_fv)

                del tracks

            print(f'Completed {file}')
            if (i+1) % 10 == 0:
                print(f'{i+1} files were completed.')

    # Save CSV files once after processing all videos
    if HOG_FISHER_VECTOR:
        save_as_csv(hog_df, 'features/hog_features.csv')
    if HOF_FISHER_VECTOR:
        save_as_csv(hof_df, 'features/hof_features.csv')
    if MBH_FISHER_VECTOR:
        save_as_csv(mbh_df, 'features/mbh_features.csv')


if __name__ == '__main__':
    main()
