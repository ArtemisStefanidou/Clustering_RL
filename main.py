import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris
from imutils import paths
import cv2
import os
import datetime
from sklearn.metrics.pairwise import euclidean_distances
import math


class GridWorldEnv(gym.Env):

    def __init__(self, render_mode=None, res=[], clusters=3):

        self.samples = res
        self.lenSamples = len(res)
        self._agent_location = None
        self._target_location = None

        self.centroids = []
        self.silhouette_score = 0
        # self.features_images = features_images  # The amount of the images
        self.clusters = clusters
        # size = len(features_images)
        # self.size = size  # The amount of the images

        # rand_list = []
        # for i in range(self.size):
        #     rand_list = np.append(rand_list, random.randint(0, self.clusters - 1))
        # self.clusters_of_images = rand_list
        #
        # self._silhouette = 0

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).

        # assign_list = [random.randint(0, clusters-1) for _ in range(len(res))]
        assign_list = [0] * len(res)

        self.clusters_of_images = assign_list

        self.observation_space = {
            "X_samples": res,
            "Clusters_assignment": assign_list
        }

        print("observation", self.observation_space)

        # We have actions, corresponding to the change of the cluster example --> 0 to 1
        self.action_space = spaces.Discrete(3)

    def euclidean_distance(self, arr1, arr2):
        if arr1.shape != arr2.shape:
            raise ValueError("Arrays must have the same shape")

        squared_distance = np.sum((arr1 - arr2) ** 2)
        distance = np.sqrt(squared_distance)
        return distance

    def reward_function(self, p_list):
        reward_list = []
        p_max = max(p_list, key=lambda p: p[0])
        for cluster in range(0, len(p_list)):
            print("p_list.index(p_max)", p_list.index(p_max))
            if cluster == p_list.index(p_max):
                if p_list[cluster][1] == 1:
                    reward_list.append(1)
                else:
                    reward_list.append(-1)
            else:
                reward_list.append(0)
        return reward_list

    def check_action(self, cluster, action):

        new_cluster = cluster + action
        if new_cluster < 0 or new_cluster > self.clusters - 1:
            return cluster
        else:
            return new_cluster

    def _get_obs(self):
        return {"X_samples": self._agent_location, "Clusters_assignment": self.clusters_of_images}

    """εδώ θα έχουμε το silhouette για το πότε τελειώνει το episode (αν το σκορ είναι πάνω από 0,5 τότε αν σταματάει)"""

    def _get_info(self):
        return {"silhouette": self._silhouette}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        """
            Choose the agent's location uniformly at random (the agent location is random combination of the clusters 
            of the photos) 
            The agent is the list of features for every image 
            An image consists of a list of features that extracted from this image
        """
        assign_list = [0] * self.lenSamples

        self.clusters_of_images = assign_list

        # # We will sample the target's location randomly until it does not coincide with the agent's location
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )

        observation = self._get_obs()
        info = self._get_info()

        self.silhouette_score = 0

        return observation, info

    def step(self, resu):

        x_sample = random.choice(resu)
        print("x_sample", x_sample)

        print("centroids", self.centroids)

        index_x = np.where(resu == x_sample)
        print("index x", index_x[0][0])

        # print("x sample", x_sample)

        self._agent_location = x_sample

        # result = result.sort()
        random_List = []
        for i in range(0, 3):
            n = random.choice(result)
            random_List.append(n)
        # print(random_List)
        self.centroids = random_List

        p_list = []
        for item in random_List:
            distance = self.euclidean_distance(x_sample, item)
            # print("distance", distance)
            function = 1 / (1 + math.exp(-distance))
            # print("math.exp(-distance)", math.exp(-distance))
            # print("function", function)
            p_i = 5 * (1 - function)
            p_list.append(p_i)

        print("p list", p_list)
        p_tmp = 0.001
        # Filter the list to numbers greater than the given_number
        filtered_numbers = [p for p in p_list if p > p_tmp]

        # Pick a random number from the filtered list
        random_p = random.choice(filtered_numbers)

        random_p_tuple = (random_p, filtered_numbers.index(random_p))

        print("random p", random_p)
        print("random p tuple", random_p_tuple)

        self.clusters_of_images[index_x[0][0]] = random_p_tuple[1]

        # print("random", random_p)
        converted_list = [(p, 1) if p == random_p else (p, 0) for p in p_list]
        # print("convert", converted_list)
        print("converted list", converted_list)

        reward = self.reward_function(converted_list)

        print("reward", reward)

        # η σταθερά που θέλω για την εξίσωση
        a = 1

        # tmp_list = []
        for i in range(0, 3):

            'όσα είναι στο ίδιο cluster'
            tmp = [resu[j] for j in range(len(self.clusters_of_images)) if self.clusters_of_images[j] == i]

            # tmp_list.append(tmp)

            if len(tmp) != 0:
                x_axes = sum([axes[0] for axes in tmp]) / len(tmp)
                y_axes = sum([axes[1] for axes in tmp]) / len(tmp)

                self.centroids[i] = (x_axes, y_axes)

        print("--------------------results----------------", self.clusters_of_images)

        self.silhouette_score = silhouette_score(resu, np.array(self.clusters_of_images))
        print("silhouette score", self.silhouette_score)

        # tmp_centroids.append(tmp_centroid)

        # print(reward)

        '''centroids'''
        non_zero_indices = [index for index, element in enumerate(reward) if element != 0]

        print(non_zero_indices)
        # # Map the action (element of {0,1,2,3}) to the direction we walk in
        # direction = self._action_to_direction[action]
        # # We use `np.clip` to make sure we don't leave the grid
        # self._agent_location = np.clip(
        #     self._agent_location + direction, 0, self.size - 1
        # )
        # # An episode is done iff the agent has reached the target
        # terminated = np.array_equal(self._agent_location, self._target_location)
        # reward = 1 if terminated else 0  # Binary sparse rewards
        # observation = self._get_obs()
        # info = self._get_info()
        #
        # if self.render_mode == "human":
        #     self._render_frame()

        return reward
        # return observation, reward, terminated, False, info


# features description -1:  Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# feature-description -2 Color Histogram
def fd_histogram(image, mask=None):
    # conver the image to HSV colors-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # COMPUTE THE COLOR HISTΟGRAM
    hist = cv2.calcHist([image], [0, 1, 2], None, [11, 11, 11], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


data = []
labels = []
times = []
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images("./traclets")))
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    start = datetime.datetime.now()
    fv_hu_moments = fd_hu_moments(image)
    fv_histogram = fd_histogram(image)
    global_feature = np.hstack([fv_histogram, fv_hu_moments])
    end = datetime.datetime.now()
    delta = end - start
    ms = int(delta.total_seconds() * 1000)
    times.append(ms)
    data.append(global_feature)
    # extract the class label from the image path and update the labels list
    label = int(imagePath.split(os.path.sep)[-2])
    labels.append(label)

# print(f'[INFO] Average feature extraction time: {sum(times) / len(times)} ms.')
# data = np.array(data)
# print("--------------------data feature extraction----------------------", data)
# labels = np.array(labels)

x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

result = np.array(list(zip(x, y)))
# # Load IRIS Dataset
# iris = load_iris()
# dataset = iris.data
class_try = GridWorldEnv(res=result, clusters=3)
print("main class_try.observation_space", class_try.observation_space)
episodes = 15
for episode in range(1, episodes + 1):
    # the episode never ends so, I didn't call reset yet
    # state = class_try.reset()
    done = False
    score = 0

    while not done:
        # to see when moves right and left
        # env.render()
        # τυχαία κάθε φορά διαλέγει είτε να πάει δεξιά είτε αριστερά άρα είτε 0 είτε 1

        # action = (random.choice([-1, +1]), random.randint(0, len(data) - 1))
        # done δείχνει αν τελείωσε το παιχνίδι ή όχι
        class_try.step(result)
        # score += reward
    # print('Episode:{} Score:{} Silhouette:{}'.format(episode, score, silhouette_s))
