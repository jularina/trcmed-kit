import math
import gpflow as gpf
from gpflow.config import (
    default_float,
    default_jitter,
)
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.kernels import Kernel
from gpflow.kernels.base import ActiveDims
from gpflow import Parameter
from gpflow.utilities import positive, to_default_float
from typing import Optional
import numpy as np
from matplotlib import pyplot as plt
import scipy
from itertools import product


def get_baseline_kernel():
    kb_per = gpf.kernels.Periodic(base_kernel=get_se_kernel_periodic(), period=24.0)
    gpf.utilities.set_trainable(kb_per.period, False)

    return kb_per + gpf.kernels.Constant()


def get_se_kernel_periodic():
    kb_se = gpf.kernels.SquaredExponential(variance=1.0, lengthscales=10.0)
    kb_se.variance.prior = tfp.distributions.HalfNormal(to_default_float(1.0))
    kb_se.lengthscales.prior = tfp.distributions.Gamma(to_default_float(10.0), to_default_float(1.0))

    return kb_se


def get_treatment_time_meal1_kernel():
    # kse = gpf.kernels.SquaredExponential(variance=1.6, lengthscales=0.3, active_dims=[0])
    # kse.variance.prior = tfp.distributions.HalfNormal(to_default_float(1.6))
    # kse.lengthscales.prior = tfp.distributions.Gamma(to_default_float(2.0), to_default_float(3.0))

    kse = gpf.kernels.SquaredExponential(variance=1.0, lengthscales=0.3, active_dims=[0])
    gpf.utilities.set_trainable(kse, False)

    return kse


def get_treatment_time_meal2_kernel():
    # kse = gpf.kernels.SquaredExponential(variance=1.2, lengthscales=0.6, active_dims=[0])
    # kse.variance.prior = tfp.distributions.HalfNormal(to_default_float(1.2))
    # kse.lengthscales.prior = tfp.distributions.Gamma(to_default_float(2.0), to_default_float(1.5))

    kse = gpf.kernels.SquaredExponential(variance=0.1, lengthscales=0.8, active_dims=[0])
    gpf.utilities.set_trainable(kse, False)

    return kse


def get_treatment_time_meal1_kernel_lfm(l):
    klfm = LFMKernel(decay=0.9, sensitivity=1.0, lengthscales=l, active_dims=[0])

    return klfm


def get_treatment_time_meal2_kernel_lfm(l):
    klfm = LFMKernel(decay=0.9, sensitivity=1.0, lengthscales=l, active_dims=[0])

    return klfm


class HierarchicalCoregion(Kernel):
    """Class for Coregion Kernel.

    Parameters:
    output_dim (int): number of output dimensions, in that case, number of meals for all patients
    rank (int): number of degrees of correlation between outputs
    num_patients (int): number of patients
    meals_patient_idx (tf.Tensor): tensor of patients indices for every meal
    beta0 (float): hyper
    beta1 (float): hyper
    sigma_raw (float): hyper
    active_dims (list): list fo active dimensions for coregionalization
   """

    def __init__(self, output_dim: int, rank: int, num_patients: int, meals_patient_idx,
                 beta0: float, beta1: float, sigma_raw: float, active_dims: Optional[ActiveDims]):

        super().__init__(active_dims=active_dims)

        self.output_dim = output_dim
        self.rank = rank
        self.num_patients = num_patients
        self.m_pidx = meals_patient_idx  # Shape equals to Mi+Mj+...
        self.m_pidx2 = meals_patient_idx  # Shape equals to Mi+Mj+...

        self.beta0 = Parameter(beta0, transform=positive())
        self.beta0_raw = Parameter(to_default_float(1.0) * tf.ones(num_patients, dtype=tf.float64),
                                  transform=positive())
        self.beta0_raw.prior = tfp.distributions.HalfNormal(to_default_float(1.0))

        self.beta1 = Parameter(beta1, transform=positive())
        self.beta1_raw = Parameter(to_default_float(1.0) * tf.ones(num_patients, dtype=tf.float64),
                                   transform=positive())
        self.beta1_raw.prior = tfp.distributions.HalfNormal(to_default_float(1.0))

        self.sigma_raw = sigma_raw

        kappa = np.ones(self.output_dim)
        self.kappa = Parameter(kappa, transform=positive())

    def set_meals_pidx(self, m_pidx):
        self.m_pidx = m_pidx

    def set_meals_pidx2(self, m_pidx2):
        self.m_pidx2 = m_pidx2

    def output_covariance(self, x, x2):
        beta0 = self.beta0 + self.sigma_raw * self.beta0_raw  # Shape equals to number of patients
        beta1 = self.beta1 + self.sigma_raw * self.beta1_raw  # Shape equals to number of patients
        beta0_vec = tf.reshape(tf.gather(beta0, self.m_pidx),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...
        beta1_vec = tf.reshape(tf.gather(beta1, self.m_pidx),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...
        W = beta0_vec+beta1_vec * x  # Shape (Mi+Mj+..., 1)

        beta0_vec = tf.reshape(tf.gather(beta0, self.m_pidx2),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...
        beta1_vec = tf.reshape(tf.gather(beta1, self.m_pidx2),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...
        W2 = beta0_vec+beta1_vec * x2  # Shape (Mi+Mj+..., 1)

        B = tf.linalg.matmul(tf.reshape(W, (-1, 1)), tf.reshape(W2, (-1, 1)), transpose_b=True)

        if tf.shape(x)[0] == tf.shape(x2)[0] and tf.shape(x)[0] == tf.shape(self.kappa)[0]:
            B = B + tf.linalg.diag(self.kappa)

        return B

    def output_variance(self, x):
        beta0 = self.beta0 + self.sigma_raw * self.beta0_raw  # Shape equals to number of patients
        beta1 = self.beta1 + self.sigma_raw * self.beta1_raw  # Shape equals to number of patients
        beta0_vec = tf.reshape(tf.gather(beta0, self.m_pidx),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...
        beta1_vec = tf.reshape(tf.gather(beta1, self.m_pidx),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...

        W = beta0_vec+beta1_vec * x  # Shape (Mi+Mj+..., 1)
        B_diag = tf.reduce_sum(tf.square(W), 1)  # As B is diagonal, the sum across rows can be computed

        if tf.shape(x)[0] == tf.shape(self.kappa)[0]:
            B_diag += self.kappa

        return B_diag

    def K(self, x, x2=None):
        m = tf.cast(x[:, 1],
                    tf.float64)  # Column with meals values for each meal-glucose entry of shape (M*NumGluc,1). It starts from 3d column of deltas.

        if x2 is None:
            m2 = m
        else:
            m2 = tf.cast(x2[:, 1], tf.float64)

        B = self.output_covariance(tf.reshape(tf.unique(m)[0], (-1, 1)),
                                   tf.reshape(tf.unique(m2)[0], (-1,
                                                                 1)))  # Selecting unique meals from m. It's equal to Mi+Mj+... Size of B=(Mi+Mj+..., Mi+Mj+...).
        x = tf.cast(x[:, 0],
                    tf.int32)  # Selecting indices of meals, corresponding to each meal-glucose pair of shape (M*NumGluc,1).

        if x2 is None:
            x2 = x
        else:
            x2 = tf.cast(x2[:, 0], tf.int32)

        return tf.gather(tf.transpose(tf.gather(tf.transpose(B), x2)),
                         x)  # Inner - shape is gathered from (Mi+Mj+..., Mi+Mj+...) to (Mi+Mj+..., M*NumGluc). Then to (M*NumGluc, M*NumGluc)

    def K_diag(self, x):
        m = tf.cast(x[:, 1], tf.float64)
        B_diag = self.output_variance(tf.reshape(tf.unique(m)[0], (-1, 1)))
        x = tf.cast(x[:, 0], tf.int32)

        return tf.gather(B_diag, x)


class HierarchicalCoregion1(Kernel):
    """Class for Coregion Kernel.

    Parameters:
    output_dim (int): number of output dimensions, in that case, number of meals for all patients
    rank (int): number of degrees of correlation between outputs
    num_patients (int): number of patients
    meals_patient_idx (tf.Tensor): tensor of patients indices for every meal
    beta0 (float): hyper
    beta1 (float): hyper
    sigma_raw (float): hyper
    active_dims (list): list fo active dimensions for coregionalization
   """

    def __init__(self, output_dim: int, rank: int, num_patients: int, meals_patient_idx,
                 beta0: float, beta1: float, sigma_raw: float, active_dims: Optional[ActiveDims]):

        super().__init__(active_dims=active_dims)

        self.output_dim = output_dim
        self.rank = rank
        self.num_patients = num_patients
        self.m_pidx = meals_patient_idx  # Shape equals to Mi+Mj+...
        self.m_pidx2 = meals_patient_idx  # Shape equals to Mi+Mj+...

        self.beta0 = Parameter(beta0, transform=positive())
        self.beta0.prior = tfp.distributions.Gamma(to_default_float(2.0), to_default_float(3.0))
        self.beta0_raw = Parameter(to_default_float(3.0) * tf.ones(num_patients, dtype=tf.float64),
                                   transform=positive())
        self.beta0_raw.prior = tfp.distributions.Gamma(to_default_float(3.0), to_default_float(3.0))

        self.beta1 = Parameter(beta1, transform=positive())
        self.beta1.prior = tfp.distributions.Gamma(to_default_float(2.0), to_default_float(3.0))
        self.beta1_raw = Parameter(to_default_float(3.0) * tf.ones(num_patients, dtype=tf.float64),
                                   transform=positive())
        self.beta1_raw.prior = tfp.distributions.Gamma(to_default_float(3.0), to_default_float(3.0))

        self.sigma_raw = Parameter(sigma_raw, transform=positive())

        kappa = np.ones(self.output_dim)
        self.kappa = Parameter(kappa, transform=positive())

    def set_meals_pidx(self, m_pidx):
        self.m_pidx = m_pidx

    def set_meals_pidx2(self, m_pidx2):
        self.m_pidx2 = m_pidx2

    def output_covariance(self, x, x2):
        beta0 = self.beta0+self.sigma_raw * self.beta0_raw  # Shape equals to number of patients
        beta1 = self.beta1+self.sigma_raw * self.beta1_raw  # Shape equals to number of patients
        beta0_vec = tf.reshape(tf.gather(beta0, self.m_pidx),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...
        beta1_vec = tf.reshape(tf.gather(beta1, self.m_pidx),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...
        W = beta0_vec+beta1_vec * x  # Shape (Mi+Mj+..., 1)

        beta0_vec = tf.reshape(tf.gather(beta0, self.m_pidx2),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...
        beta1_vec = tf.reshape(tf.gather(beta1, self.m_pidx2),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...
        W2 = beta0_vec+beta1_vec * x2  # Shape (Mi+Mj+..., 1)

        B = tf.linalg.matmul(tf.reshape(W, (-1, 1)), tf.reshape(W2, (-1, 1)), transpose_b=True)

        if tf.shape(x)[0] == tf.shape(x2)[0] and tf.shape(x)[0] == tf.shape(self.kappa)[0]:
            B = B + tf.linalg.diag(self.kappa)

        return B

    def output_variance(self, x):
        beta0 = self.beta0+self.sigma_raw * self.beta0_raw  # Shape equals to number of patients
        beta1 = self.beta1+self.sigma_raw * self.beta1_raw  # Shape equals to number of patients
        beta0_vec = tf.reshape(tf.gather(beta0, self.m_pidx),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...
        beta1_vec = tf.reshape(tf.gather(beta1, self.m_pidx),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...

        W = beta0_vec+beta1_vec * x  # Shape (Mi+Mj+..., 1)
        B_diag = tf.reduce_sum(tf.square(W), 1)  # As B is diagonal, the sum across rows can be computed

        if tf.shape(x)[0] == tf.shape(self.kappa)[0]:
            B_diag += self.kappa

        return B_diag

    def K(self, x, x2=None):
        m = tf.cast(x[:, 1],
                    tf.float64)  # Column with meals values for each meal-glucose entry of shape (M*NumGluc,1). It starts from 3d column of deltas.

        if x2 is None:
            m2 = m
        else:
            m2 = tf.cast(x2[:, 1], tf.float64)

        B = self.output_covariance(tf.reshape(tf.unique(m)[0], (-1, 1)),
                                   tf.reshape(tf.unique(m2)[0], (-1,
                                                                 1)))  # Selecting unique meals from m. It's equal to Mi+Mj+... Size of B=(Mi+Mj+..., Mi+Mj+...).
        x = tf.cast(x[:, 0],
                    tf.int32)  # Selecting indices of meals, corresponding to each meal-glucose pair of shape (M*NumGluc,1).

        if x2 is None:
            x2 = x
        else:
            x2 = tf.cast(x2[:, 0], tf.int32)

        return tf.gather(tf.transpose(tf.gather(tf.transpose(B), x2)),
                         x)  # Inner - shape is gathered from (Mi+Mj+..., Mi+Mj+...) to (Mi+Mj+..., M*NumGluc). Then to (M*NumGluc, M*NumGluc)

    def K_diag(self, x):
        m = tf.cast(x[:, 1], tf.float64)
        B_diag = self.output_variance(tf.reshape(tf.unique(m)[0], (-1, 1)))
        x = tf.cast(x[:, 0], tf.int32)

        return tf.gather(B_diag, x)


class LFMKernel(Kernel):
    """Class for the LFM Kernel, derived on the basis of Squared Exponential.

    Parameters:
    decay (float): decay rate for the ODE
    sensitivity (float): sensitivity term for the ODE
    active_dims (list): list fo active dimensions for meals times differences
   """

    def __init__(self, decay: float, sensitivity: float, lengthscales: float, active_dims: Optional[ActiveDims]):
        super().__init__(active_dims=active_dims)

        self.d = Parameter(decay, transform=positive())
        self.s = Parameter(sensitivity, transform=positive())
        self.d.prior = tfp.distributions.Gamma(to_default_float(2.0), to_default_float(1.0))
        self.s.prior = tfp.distributions.Gamma(to_default_float(2.0), to_default_float(1.0))

        self.l = Parameter(lengthscales, transform=positive())

    def K(self, x, x2=None):
        if x2 is None:
            x2 = x

        # # Prepare data
        #x_td = tf.reshape(tf.cast(x[:, 0], tf.float64), (-1,1)) # N
        # x_mi = tf.reshape(tf.cast(x[:, 1], tf.float64), (-1, 1))  # N
        # x_mi_unique = tf.unique(x[:, 1])[0]
        # x_glt = tf.reshape(tf.cast(x[:, 2], tf.float64), (-1,1))
        # x_mt = tf.reshape(tf.cast(x[:, 3], tf.float64), (-1,1))
        #x2_td = tf.reshape(tf.cast(x2[:, 0], tf.float64), (-1,1)) # M
        # x2_mi = tf.reshape(tf.cast(x2[:, 1], tf.float64), (-1, 1))  # M
        # x2_mi_unique = tf.unique(x2[:, 1])[0]
        # x2_glt = tf.reshape(tf.cast(x2[:, 2], tf.float64),  (-1,1))
        # x2_mt = tf.reshape(tf.cast(x2[:, 3], tf.float64),  (-1,1))

        ########## FIRST EQUATION ##########
        # K = tf.zeros([x_td.shape[0],x_td.shape[0]])
        # for m in x_mi_unique:
        #     mask = (x_mi == m)
        #     mask_indices = tf.where(mask)[:,0].numpy()
        #     x = tf.reshape(x_td[mask], (-1,1))
        #     x2 = tf.reshape(x2_td[mask], (-1,1))

        # x = tf.reshape(tf.cast(x[:, 0], tf.float64), (-1, 1))  # N
        # x2 = tf.reshape(tf.cast(x2[:, 0], tf.float64), (-1, 1))  # N
        #
        # coeff1 = tf.math.square(self.s) * math.sqrt(math.pi) * self.l / 2
        # gamma = self.d * self.l / 2
        # coeff2 = tf.math.exp(tf.math.square(gamma))/(2*self.d)
        #
        # # Left part
        # deltas = x - tf.transpose(x2)
        # a1 = tf.math.exp(-1 * self.d * deltas)
        # a2 = tf.math.exp(-1 * self.d * (x + 1))
        # b1 = tf.math.erf((deltas / self.l) - gamma) + tf.transpose(tf.math.erf((x2 / self.l) + gamma))
        # b2 = tf.math.erf((x / self.l) - gamma) + tf.math.erf(gamma)
        # lp = a1 * b1 - a2 * b2
        #
        # # Right part
        # deltas = x2 - tf.transpose(x)
        # a1 = tf.math.exp(-1 * self.d * deltas)
        # a2 = tf.math.exp(-1 * self.d * (x2 + 1))
        # b1 = tf.math.erf((deltas / self.l) - gamma) + tf.transpose(tf.math.erf((x / self.l) + gamma))
        # b2 = tf.math.erf((x2 / self.l) - gamma) + tf.math.erf(gamma)
        # rp = a1 * b1 - a2 * b2
        #
        # K = coeff1 * coeff2 * (lp + tf.transpose(rp))

            # # Adding elements
            # for i in mask_indices:
            #     for j in mask_indices:
            #         upd = tf.reshape(tf.cast(K1[i-min(mask_indices),j-min(mask_indices)], tf.float32), [-1])
            #         idx = tf.constant([[i,j]])
            #         K = tf.tensor_scatter_nd_add(K, idx, upd)

        # ########### SECOND EQUATION - G4 ##########
        K = self.k2(x,x2)

        # figure = plt.figure()
        # axes = figure.add_subplot(111)
        # caxes = axes.matshow(K, interpolation='nearest')
        # figure.colorbar(caxes)
        # plt.show()

        return K

    def K_diag(self, x):
        # coeff1 = tf.math.square(self.s) * math.sqrt(math.pi) * self.l / 2
        # gamma = self.d * self.l / 2
        # coeff2 = tf.math.exp(tf.math.square(gamma))/(2*self.d)
        #
        # # Left part
        # deltas = x - tf.transpose(x)
        # a1 = tf.math.exp(-1 * self.d * deltas)
        # a2 = tf.math.exp(-1 * self.d * (x + 1))
        # b1 = tf.math.erf((deltas / self.l) - gamma) + tf.transpose(tf.math.erf((x / self.l) + gamma))
        # b2 = tf.math.erf((x / self.l) - gamma) + tf.math.erf(gamma)
        # lp = a1 * b1 - a2 * b2
        #
        # B = coeff1 * coeff2 * (lp + tf.transpose(lp))
        #
        # B = tf.linalg.diag_part(B)

        B = tf.linalg.diag_part(self.K(x))

        return B

    # Cheng et al kernel for t>tm, t_prime>tm
    def k(self, t, t_prime, tm=0.0):
        vk, vj = self.l * self.d / 2, self.l * self.d / 2

        # Eq. 27 - only is non-zero
        hjk = (np.exp(vj ** 2)) / (self.d + self.d) * tf.math.exp(-self.d * t) * (
                    tf.math.exp(self.d * t_prime) * (tf.math.erf((t - t_prime) / self.l - vj) + tf.math.erf((t_prime - tm) / self.l + vj)) - tf.math.exp(
                self.d * tm) * tf.math.exp(self.d * tm - self.d * t_prime) * (tf.math.erf((t - tm) / self.l - vj) + tf.math.erf(vj)))
        hkj = (tf.math.exp(vk ** 2)) / (self.d + self.d) * tf.math.exp(-self.d * t_prime) * (
                    tf.math.exp(self.d * t) * (tf.math.erf((t_prime - t) / self.l - vk) + tf.math.erf((t - tm) / self.l + vk)) - tf.math.exp(self.d * tm) * tf.math.exp(
                self.d * tm - self.d * t) * (tf.math.erf((t_prime - tm) / self.l - vk) + tf.math.erf(vk)))
        g4 = (self.s * self.s * np.sqrt(np.pi) * self.l) / 2 * (hjk + hkj)

        # Eq. 23
        kjk = g4

        return kjk

    # Cheng et al kernel for t>tm, t_prime>tm
    def k2(self, t, t_prime, tm=0.0):
        vk, vj = self.l * self.d / 2, self.l * self.d / 2

        # Eq. 27 - only is non-zero
        deltas = t - tf.transpose(t_prime)
        hjk = (tf.math.exp(vj ** 2)) / (self.d + self.d) * tf.math.exp(-self.d * t) * (
                    tf.transpose(tf.math.exp(self.d * t_prime)) * (tf.math.erf((deltas) / self.l - vj) + tf.transpose(tf.math.erf((t_prime - tm) / self.l + vj))) - tf.math.exp(
                self.d * tm) * tf.transpose(tf.math.exp(self.d * tm - self.d * t_prime)) * (tf.math.erf((t - tm) / self.l - vj) + tf.math.erf(vj)))

        deltas = t_prime-tf.transpose(t)
        hkj = (tf.math.exp(vk ** 2)) / (self.d + self.d) * tf.math.exp(-self.d * t_prime) * (
                    tf.transpose(tf.math.exp(self.d * t)) * (tf.math.erf((deltas) / self.l - vk) + tf.transpose(tf.math.erf((t - tm) / self.l + vk))) - tf.math.exp(self.d * tm) * tf.transpose(tf.math.exp(
                self.d * tm - self.d * t)) * (tf.math.erf((t_prime - tm) / self.l - vk) + tf.math.erf(vk)))


        g4 = (self.s * self.s * np.sqrt(np.pi) * self.l) / 2 * (hjk + tf.transpose(hkj))

        # Eq. 23
        kjk = g4

        return kjk


class LFMLatentKernel(Kernel):
    def __init__(self, decay: float, sensitivity: float, lengthscales: float, active_dims: Optional[ActiveDims]):
        super().__init__(active_dims=active_dims)

        self.d = Parameter(decay, transform=positive())
        self.s = Parameter(sensitivity, transform=positive())
        self.d.prior = tfp.distributions.Gamma(to_default_float(2.0), to_default_float(1.0))
        self.s.prior = tfp.distributions.Gamma(to_default_float(2.0), to_default_float(1.0))

        self.l = Parameter(lengthscales, transform=positive())

    def K(self, x, x2=None):
        if x2 is None:
            x2 = x
        K = self.k2(x,x2)

        return K

    def K_diag(self, x):

        B = tf.linalg.diag_part(self.K(x))

        return B

    # Cheng et al kernel for t>tm, t_prime>tm
    def k2(self, t, t_prime, tm=0.0):
        v = self.l * self.d / 2

        h1 = self.s * tf.math.exp(-self.d*t) * tf.transpose(tf.math.exp(-((t_prime-tm)/self.l)**2)) * 1/self.d * (tf.math.exp(self.d*tm)-1)

        deltas = t - tf.transpose(t_prime)
        h2 = self.s * (np.sqrt(np.pi) * self.l) / 2 * tf.math.exp(-self.d*(deltas)) * tf.math.exp(v ** 2) * (tf.math.erf((deltas) / self.l - v) + tf.transpose(tf.math.erf((t_prime - tm) / self.l + v)))

        kjk = h1 + h2

        return kjk


######### Kernels for artificial data ##################
class LFMKernelToy(Kernel):
    """Class for the LFM Kernel, derived on the basis of Squared Exponential.

    Parameters:
    decay (float): decay rate for the ODE
    sensitivity (float): sensitivity term for the ODE
    active_dims (list): list fo active dimensions for meals times differences
   """

    def __init__(self, decay: float, sensitivity: float, lengthscales: float, t_meals, active_dims: Optional[ActiveDims]):
        super().__init__(active_dims=active_dims)

        self.d = Parameter(decay, transform=positive())
        self.s = Parameter(sensitivity, transform=positive())
        self.d.prior = tfp.distributions.Gamma(to_default_float(2.0), to_default_float(1.0))
        self.s.prior = tfp.distributions.Gamma(to_default_float(2.0), to_default_float(1.0))
        self.l = Parameter(lengthscales, transform=positive())
        self.ts_m = Parameter(t_meals, dtype=default_float(), trainable=False)

    def K(self, x, x2=None):
        if x2 is None:
            x2 = x

        K = np.zeros((len(x), len(x2)))
        for t_m in self.ts_m:
            # For x
            t_masked = x[x > t_m]
            t_first = list(x).index(t_masked[0])
            t_masked_idx = np.arange(len(t_masked)) + t_first

            # For x2
            t_masked2 = x2[x2 > t_m]
            t_first2 = list(x2).index(t_masked2[0])
            t_masked_idx2 = np.arange(len(t_masked2)) + t_first2

            # Product of indices
            prod = list(product(t_masked, t_masked2))
            prod_idx = list(product(t_masked_idx, t_masked_idx2))

            for p, p_idx in zip(prod, prod_idx):
                t, t_prime = p[0], p[1]
                t_idx, t_prime_idx = p_idx[0], p_idx[1]
                kjk = self.k(t - t_m, t_prime - t_m)
                K[t_idx, t_prime_idx] += kjk

        # figure = plt.figure()
        # axes = figure.add_subplot(111)
        # caxes = axes.matshow(K, interpolation='nearest')
        # figure.colorbar(caxes)
        # plt.show()

        return K

    def K_diag(self, x):
        B = tf.linalg.diag_part(self.K(x))

        return B

    # Cheng et al kernel for t>tm, t_prime>tm
    def k(self, t, t_prime, tm=0.0):
        vk, vj = self.l * self.d / 2, self.l * self.d / 2

        # Eq. 27 - only is non-zero
        hjk = (np.exp(vj ** 2)) / (self.d + self.d) * tf.math.exp(-self.d * t) * (
                    tf.math.exp(self.d * t_prime) * (tf.math.erf((t - t_prime) / self.l - vj) + tf.math.erf((t_prime - tm) / self.l + vj)) - tf.math.exp(
                self.d * tm) * tf.math.exp(self.d * tm - self.d * t_prime) * (tf.math.erf((t - tm) / self.l - vj) + tf.math.erf(vj)))
        hkj = (tf.math.exp(vk ** 2)) / (self.d + self.d) * tf.math.exp(-self.d * t_prime) * (
                    tf.math.exp(self.d * t) * (tf.math.erf((t_prime - t) / self.l - vk) + tf.math.erf((t - tm) / self.l + vk)) - tf.math.exp(self.d * tm) * tf.math.exp(
                self.d * tm - self.d * t) * (tf.math.erf((t_prime - tm) / self.l - vk) + tf.math.erf(vk)))
        g4 = (self.s * self.s * np.sqrt(np.pi) * self.l) / 2 * (hjk + hkj)

        # Eq. 23
        kjk = g4

        return kjk


class LFMOutputKernelToy(Kernel):
    def __init__(self, decay_rate, sensitivity, treatment_time, lengthscale):
        super().__init__()
        self.d = tf.convert_to_tensor(decay_rate, dtype=default_float())
        self.s = tf.convert_to_tensor(sensitivity, dtype=default_float())
        self.t_m = tf.convert_to_tensor(treatment_time, dtype=default_float())
        self.l = tf.convert_to_tensor(lengthscale, dtype=default_float())

    def K(self, x: tf.Tensor, x2: tf.Tensor = None):
        if x2 is None:
            x2 = x

        K = self.k(x,x2)

        return K

    def k(self, t, t_prime):
        v = self.l * self.d / 2
        k = (np.sqrt(np.pi) * self.l) / 2 * tf.math.exp(v ** 2) * self.s * tf.math.exp(
                -1 * self.d * (t-t_prime)) * (tf.math.erf((t-t_prime)/self.l - v)+tf.math.erf((t_prime-self.t_m)/self.l + v))

        return k

    def K_diag(self, x):
        return tf.linalg.diag_part(self.K(x))


class TimeMarkedSquaredExponential(Kernel):
    def __init__(self, t_m, lengthscale):
        super().__init__()
        self.t_m = Parameter(t_m, dtype=default_float())
        self.l = Parameter(lengthscale, dtype=default_float())

    @tf.function
    def K(self, x: tf.Tensor, x2: tf.Tensor = None):
        if x2 is None:
            x2 = x

        K = self.k(x, x2)

        return K

    def k(self, t, t_prime):
        return tf.exp(-((t - self.t_m) - (t_prime - self.t_m)) ** 2 / (self.l**2))

    def K_diag(self, x):

        return tf.linalg.diag_part(self.K(x))