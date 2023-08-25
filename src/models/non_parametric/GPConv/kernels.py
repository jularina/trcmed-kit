import math
import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.kernels import Kernel
from gpflow.kernels.base import ActiveDims
from gpflow import Parameter
from gpflow.utilities import positive, to_default_float
from typing import Optional
import numpy as np
from matplotlib import pyplot as plt
import  scipy


def get_baseline_kernel():
    kb_per = gpf.kernels.Periodic(base_kernel=get_se_kernel_periodic(), period=24.0)
    gpf.utilities.set_trainable(kb_per.period, False)

    return kb_per + gpf.kernels.Constant()


def get_se_kernel_periodic():
    kb_se = gpf.kernels.SquaredExponential(variance=1.0, lengthscales=10.0)
    kb_se.variance.prior = tfp.distributions.HalfNormal(to_default_float(1.0))
    kb_se.lengthscales.prior = tfp.distributions.Gamma(to_default_float(10.0), to_default_float(1.0))

    return kb_se


def get_treatment_time_meal1_kernel_lfm():
    klfm = LFMKernel(l=1.0, l1=1.0, l2=1.0, active_dims=[0, 3])

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
                 beta1: float, beta11: float, sigma_raw: float, active_dims: Optional[ActiveDims]):

        super().__init__(active_dims=active_dims)

        self.output_dim = output_dim
        self.rank = rank
        self.num_patients = num_patients
        self.m_pidx = meals_patient_idx  # Shape equals to Mi+Mj+...
        self.m_pidx2 = meals_patient_idx  # Shape equals to Mi+Mj+...

        self.beta1 = Parameter(beta1, transform=positive())
        self.beta1.prior = tfp.distributions.Gamma(to_default_float(3.0), to_default_float(5.0))
        self.beta1_raw = Parameter(to_default_float(4.0) * tf.ones(num_patients, dtype=tf.float64),
                                   transform=positive())
        self.beta1_raw.prior = tfp.distributions.Gamma(to_default_float(8.0), to_default_float(3.0))

        self.beta11 = Parameter(beta11, transform=positive())
        self.beta11.prior = tfp.distributions.Gamma(to_default_float(3.0), to_default_float(5.0))
        self.beta11_raw = Parameter(to_default_float(4.5) * tf.ones(num_patients, dtype=tf.float64),
                                   transform=positive())
        self.beta11_raw.prior = tfp.distributions.Gamma(to_default_float(8.0), to_default_float(3.0))

        self.sigma_raw = sigma_raw

        kappa = np.ones(self.output_dim)
        self.kappa = Parameter(kappa, transform=positive())

    def set_meals_pidx(self, m_pidx):
        self.m_pidx = m_pidx

    def set_meals_pidx2(self, m_pidx2):
        self.m_pidx2 = m_pidx2

    def output_covariance(self, x, x2):
        beta1 = self.beta1 + self.sigma_raw * self.beta1_raw  # Shape equals to number of patients
        beta1_vec = tf.reshape(tf.gather(beta1, self.m_pidx),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...
        beta11 = self.beta11 + self.sigma_raw * self.beta11_raw  # Shape equals to number of patients
        beta11_vec = tf.reshape(tf.gather(beta11, self.m_pidx),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...

        W = beta1_vec * tf.reshape(x[:, 0], (-1,1)) + beta11_vec * tf.reshape(x[:, 1], (-1,1))  # Shape (Mi+Mj+..., 1)
        #W = tf.concat([beta1_vec * tf.reshape(x[:, 0], (-1,1)), beta11_vec * tf.reshape(x[:, 1], (-1,1))], axis=1)

        beta1_vec = tf.reshape(tf.gather(beta1, self.m_pidx2),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...
        beta11_vec = tf.reshape(tf.gather(beta11, self.m_pidx2),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...
        W2 = beta1_vec * tf.reshape(x2[:, 0], (-1,1)) + beta11_vec * tf.reshape(x2[:, 1], (-1,1)) # Shape (Mi+Mj+..., 1)
        #W2 = tf.concat([beta1_vec * tf.reshape(x2[:, 0], (-1,1)), beta11_vec * tf.reshape(x2[:, 1], (-1,1))], axis=1)

        B = tf.linalg.matmul(tf.reshape(W, (-1, 1)), tf.reshape(W2, (-1, 1)), transpose_b=True)
        #B = tf.linalg.matmul(W, W2, transpose_b=True)

        if tf.shape(x)[0] == tf.shape(x2)[0] and tf.shape(x)[0] == tf.shape(self.kappa)[0]:
            B = B + tf.linalg.diag(self.kappa)

        return B

    def output_variance(self, x):
        beta1 = self.beta1 + self.sigma_raw * self.beta1_raw  # Shape equals to number of patients
        beta1_vec = tf.reshape(tf.gather(beta1, self.m_pidx),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...
        beta11 = self.beta11 + self.sigma_raw * self.beta11_raw  # Shape equals to number of patients
        beta11_vec = tf.reshape(tf.gather(beta11, self.m_pidx),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...

        W = beta1_vec * tf.reshape(x[:, 0], (-1,1)) + beta11_vec * tf.reshape(x[:, 1], (-1,1)) # Shape (Mi+Mj+..., 1)
        B_diag = tf.reduce_sum(tf.square(W), 1)  # As B is diagonal, the sum across rows can be computed
        #W = tf.concat([beta1_vec * tf.reshape(x[:, 0], (-1, 1)), beta11_vec * tf.reshape(x[:, 1], (-1, 1))], axis=1)
        #B_diag = tf.reduce_sum(tf.linalg.matmul(W, W, transpose_b=True), 1)

        if tf.shape(x)[0] == tf.shape(self.kappa)[0]:
            B_diag += self.kappa

        return B_diag

    def K(self, x, x2=None):
        # m = tf.cast(x[:, 1],
        #             tf.float64)  # Column with meals values for each meal-glucose entry of shape (M*NumGluc,1). It starts from 3d column of deltas.
        #
        # if x2 is None:
        #     m2 = m
        # else:
        #     m2 = tf.cast(x2[:, 1], tf.float64)
        #
        # B = self.output_covariance(tf.reshape(tf.unique(m)[0], (-1, 1)),
        #                            tf.reshape(tf.unique(m2)[0], (-1,1)))  # Selecting unique meals from m. It's equal to Mi+Mj+... Size of B=(Mi+Mj+..., Mi+Mj+...).
        # x = tf.cast(x[:, 0],
        #             tf.int32)  # Selecting indices of meals, corresponding to each meal-glucose pair of shape (M*NumGluc,1).
        #
        # if x2 is None:
        #     x2 = x
        # else:
        #     x2 = tf.cast(x2[:, 0], tf.int32)
        # k = tf.gather(tf.transpose(tf.gather(tf.transpose(B), x2)),
        #                  x)  # Inner - shape is gathered from (Mi+Mj+..., Mi+Mj+...) to (Mi+Mj+..., M*NumGluc). Then to (M*NumGluc, M*NumGluc)

        x_unique = tf.raw_ops.UniqueV2(x=x, axis=[0])[0]
        m = tf.cast(x_unique[:, 1:], tf.float64)  # Column with meals values for each meal-glucose entry of shape (M*NumGluc,1). It starts from 3d column of deltas.

        if x2 is None:
            m2 = m
        else:
            x2_unique = tf.raw_ops.UniqueV2(x=x2, axis=[0])[0]
            m2 = tf.cast(x2_unique[:, 1:], tf.float64)

        B = self.output_covariance(tf.reshape(m, (-1, 2)),
                                   tf.reshape(m2, (-1, 2)))  # Selecting unique meals from m. It's equal to Mi+Mj+... Size of B=(Mi+Mj+..., Mi+Mj+...).
        x = tf.cast(x[:, 0], tf.int32)  # Selecting indices of meals, corresponding to each meal-glucose pair of shape (M*NumGluc,1).

        if x2 is None:
            x2 = x
        else:
            x2 = tf.cast(x2[:, 0], tf.int32)

        k = tf.gather(tf.transpose(tf.gather(tf.transpose(B), x2)), x)

        return k
    def K_diag(self, x):

        m = tf.cast(x[:, 1:], tf.float64)
        B_diag = self.output_variance(tf.reshape(tf.raw_ops.UniqueV2(x=m, axis=[0])[0], (-1, 2)))
        x = tf.cast(x[:, 0], tf.int32)

        return tf.gather(B_diag, x)


class LFMKernel(Kernel):
    """Class for the Convolution Kernel.
   """

    def __init__(self, num_patients: int, meals_patient_idx, active_dims: Optional[ActiveDims]):
        super().__init__(active_dims=active_dims)

        self.num_patients = num_patients
        self.m_pidx = meals_patient_idx  # Shape equals to Mi+Mj+...
        self.m_pidx2 = meals_patient_idx  # Shape equals to Mi+Mj+...

        self.l = Parameter(to_default_float(0.4), transform=positive())
        self.l.prior = tfp.distributions.Gamma(to_default_float(2.0), to_default_float(3.0))

        self.beta_raw = Parameter(to_default_float(0.001) * tf.ones(num_patients, dtype=tf.float64),
                                   transform=positive())
        #self.beta_raw.prior = tfp.distributions.Gamma(to_default_float(1.0), to_default_float(3.0))
        self.alpha_raw = Parameter(to_default_float(0.01) * tf.ones(num_patients, dtype=tf.float64),
                                  transform=positive())

    def set_meals_pidx(self, m_pidx):
        self.m_pidx = m_pidx

    def set_meals_pidx2(self, m_pidx2):
        self.m_pidx2 = m_pidx2

    def k(self, t, t_prime, m, m2, idx, idx2):
        # Scalers for the first meal
        # l = tf.reshape(tf.gather(self.l, self.m_pidx), (-1, 1))
        # l = tf.reshape(tf.gather(l, idx), (-1, 1))
        alpha_vec = tf.reshape(tf.gather(self.alpha_raw, self.m_pidx),(-1, 1))
        alpha_vec = tf.reshape(tf.gather(alpha_vec, idx), (-1, 1))
        beta_vec = tf.reshape(tf.gather(self.beta_raw, self.m_pidx),(-1, 1))
        beta_vec = tf.reshape(tf.gather(beta_vec, idx), (-1, 1))
        l1_vec = 0.01 + beta_vec*m
        m1_vec = 0.001 + alpha_vec*m

        # Scalers for the second meal
        # l2 = tf.reshape(tf.gather(self.l, self.m_pidx2), (-1, 1))
        # l2 = tf.reshape(tf.gather(l2, idx2), (1, -1))
        alpha_vec = tf.reshape(tf.gather(self.alpha_raw, self.m_pidx2),(-1, 1))
        alpha_vec = tf.reshape(tf.gather(alpha_vec, idx2), (-1, 1))
        beta_vec = tf.reshape(tf.gather(self.beta_raw, self.m_pidx2),(-1, 1))
        beta_vec = tf.reshape(tf.gather(beta_vec, idx2), (-1, 1))
        l2_vec = 0.01 + beta_vec * m2
        m2_vec = 0.001 + alpha_vec*m2

        # Lengthscale
        # ls = tf.math.multiply(l,l2)
        # Lsum = l1_vec + tf.transpose(l2_vec) + ls
        Lsum = l1_vec + tf.transpose(l2_vec) + self.l
        #Lsum = self.l
        coeff = (self.l ** (1 / 2)) / (Lsum) ** (1 / 2)
        deltas = t - tf.transpose(t_prime)

        k = coeff * tf.math.exp(-1 / 2 * ((deltas-m1_vec+tf.transpose(m2_vec))**2) * ((Lsum) ** (-1)))

        return k

    def K(self, x, x2=None):

        m = tf.cast(x[:, 2], tf.float64)
        if x2 is None:
            m2 = m
        else:
            m2 = tf.cast(x2[:, 2], tf.float64)

        idx = tf.cast(x[:, 1], tf.int32)
        if x2 is None:
            idx2 = idx
        else:
            idx2 = tf.cast(x2[:, 1], tf.int32)

        x = tf.cast(x[:, 0], tf.float64)
        if x2 is None:
            x2 = x
        else:
            x2 = tf.cast(x2[:, 0], tf.float64)

        K = self.k(tf.reshape(x, (-1, 1)), tf.reshape(x2, (-1, 1)), tf.reshape(m, (-1, 1)), tf.reshape(m2, (-1, 1)), idx, idx2)

        # figure = plt.figure()
        # axes = figure.add_subplot(111)
        # caxes = axes.matshow(K, interpolation='nearest')
        # figure.colorbar(caxes)
        # plt.show()
        #
        # t = (K == tf.transpose(K))

        return K

    def K_diag(self, x):

        B = tf.linalg.diag_part(self.K(x))

        return B