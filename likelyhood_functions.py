import numpy as np
import pandas as pd
import os
import emcee
import multiprocessing
import scipy.integrate as integrate
import math
import matplotlib.pyplot as plt
import corner
plt.rcParams.update({'font.size': 12})
from tqdm import tqdm
import scipy.linalg as la

# Define file paths
BASE_DIRECTORY = 'D:\\OneDrive\\Documents\\progetto_chiura\\mcmc\\save'
COVARIANCE_DIRECTORY = 'D:\\OneDrive\\Documents\\progetto_chiura\\mcmc\\save\\covariance_matrix'

# Define constants
G = 4.30091e-6

# Define directories and variables
STAR_NUMBER = 100
ANISOTROPY_TYPE = 'iso'
PROFILE_TYPE = 'core'
SPACE_TYPE = 'cartesian'


NWALKER = 50 #50
NDIM = 3 #this dfines the number of model parameters that required to be constrained.
NITER = 500 #500

SIGMA_V = 0  # Set this to 0, 1, or 5 as needed


#Create mockdata
def load_data(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, names=['x', 'y', 'z', 'vx', 'vy', 'vz'])

    return data.to_numpy()  # Convert the DataFrame to a NumPy array, 
        
#Create cov matrixes
def load_covariance_matrix(star_index):


    file_name = f'Cij_cartesian_stars_{star_index}_{PROFILE_TYPE}_beta_{ANISOTROPY_TYPE}.txt'
    covariance_star_path = file_path = os.path.join(COVARIANCE_DIRECTORY, f'star{STAR_NUMBER}')
    file_path = os.path.join(covariance_star_path, file_name)
    
    matrix = np.loadtxt(file_path)
    return matrix


#Load the mock data
mock_data_file = f'Mockdata_{SPACE_TYPE}_{STAR_NUMBER}_stars_{PROFILE_TYPE}_beta_{ANISOTROPY_TYPE}.txt'
mock_data_path = os.path.join(BASE_DIRECTORY, mock_data_file)
MOCK_DATA = load_data(mock_data_path) #array [][] che contiene i dati


COVARIANCE_MATRICES = []

for star_index in range(STAR_NUMBER):
    matrix = load_covariance_matrix(star_index)
    COVARIANCE_MATRICES.append(matrix) #array di matrici di covarianza

# Calculate the sigma_vt^2 value for a given covariance matrix index
def calculate_sigma_vt2(matrix_index): #matrix index gli passero poi star_index
    
    covariance_matrix = COVARIANCE_MATRICES[matrix_index]
    
    sigma_vtheta_vtheta = covariance_matrix[1, 1]
    sigma_vphi_vphi = covariance_matrix[2, 2]
    
    sigma_vt2 = (sigma_vtheta_vtheta + sigma_vphi_vphi) / 2
    return sigma_vt2

#Create a 3x3 instrumental covariance matrix with specified sigma_v.
def create_S_instrumental():
    S_instrumental = np.zeros((3, 3))
    np.fill_diagonal(S_instrumental, SIGMA_V**2)
    return S_instrumental

def calculate_mu_v():
    
    avg_vx = np.mean(MOCK_DATA[3])
    avg_vy = np.mean(MOCK_DATA[4])
    avg_vz = np.mean(MOCK_DATA[5])
    
    mu_v = np.array([avg_vx, avg_vy, avg_vz])

    return mu_v


def integrand(r, rho0, rs, gamma):
        return  r**2 * (rho0 / (((r / rs)**gamma) * (1 + r / rs)**(3 - gamma)))


def DM_profile_model(params, star_index):
    # Unpack the parameters
    rho0, rs, gamma = params

    b = np.sqrt(MOCK_DATA[star_index][0]**2 + MOCK_DATA[star_index][1]**2 + MOCK_DATA[star_index][2]**2)

    integral, _ = integrate.quad(integrand, 0, b, args = (10**rho0, 10**rs, gamma))

    V_squared = (4 * np.pi * G / b) * abs(integral)
        
    V_r = np.sqrt(V_squared)
    
    return V_r



def p_v_given_r(star_index, covariance_matrix, params):

    V_r = DM_profile_model(params, star_index) #eh ma sta velocità è un numero non un vettore
    V_r_vett = [V_r, V_r, V_r]
    
    # Calculate relative velocity vector
    # v_relative = V_r - np.linalg.norm(mu_v)
    v_relative = V_r_vett - MU_V

    # Calculate S_total which is the sum of S_instrumental and covariance_matrix
    S_total = S_INSTRUMENTAL + covariance_matrix

    log_numerator = (-0.5 * np.dot(np.dot(np.transpose(v_relative), np.linalg.inv(S_total)), v_relative))
    log_denominator = 0.5*np.log10(((2 * np.pi)**NDIM)*(np.linalg.det(S_total)))
    return log_numerator - log_denominator



#Likelyhood
def log_likelyhood(params):
    L_params = 0

    for star_index in range(STAR_NUMBER):
    
        covariance_matrix = COVARIANCE_MATRICES[star_index]
        position_cartesian = MOCK_DATA[star_index][:3]
        #v_cartesian = mock_data[star_index][3:]

        prob_given_r = p_v_given_r(star_index, covariance_matrix, params)

        L_params += prob_given_r
    if not math.isnan(L_params):
        return -np.inf
    return L_params

MU_V = calculate_mu_v()
S_INSTRUMENTAL = create_S_instrumental()