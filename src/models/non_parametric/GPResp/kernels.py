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


def get_treatment_time_meal1_kernel_lfm():
    klfm = LFMKernel(decay=0.5, sensitivity=1.0, variance=1.0, lengthscales=0.3, active_dims=[0])

    return klfm


def get_treatment_time_meal2_kernel_lfm():
    klfm = LFMKernel(decay=0.5, sensitivity=1.0, variance=0.1, lengthscales=0.8, active_dims=[0])

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
        self.beta1.prior = tfp.distributions.Gamma(to_default_float(4.0), to_default_float(3.0))
        self.beta1_raw = Parameter(to_default_float(3.0) * tf.ones(num_patients, dtype=tf.float64),
                                   transform=positive())
        self.beta1_raw.prior = tfp.distributions.Gamma(to_default_float(4.0), to_default_float(3.0))

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
        #W = beta0_vec+beta1_vec * x  # Shape (Mi+Mj+..., 1)
        W = beta1_vec * x  # Shape (Mi+Mj+..., 1)

        beta0_vec = tf.reshape(tf.gather(beta0, self.m_pidx2),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...
        beta1_vec = tf.reshape(tf.gather(beta1, self.m_pidx2),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...
        #W2 = beta0_vec+beta1_vec * x2  # Shape (Mi+Mj+..., 1)
        W2 = beta1_vec * x2  # Shape (Mi+Mj+..., 1)

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

        #W = beta0_vec+beta1_vec * x  # Shape (Mi+Mj+..., 1)
        W = beta1_vec * x  # Shape (Mi+Mj+..., 1)
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
        Bnew = tf.gather(tf.transpose(B), x2)
        return tf.gather(tf.transpose(tf.gather(tf.transpose(B), x2)),
                         x)  # Inner - shape is gathered from (Mi+Mj+..., Mi+Mj+...) to (Mi+Mj+..., M*NumGluc). Then to (M*NumGluc, M*NumGluc)

    def K_diag(self, x):
        m = tf.cast(x[:, 1], tf.float64)
        B_diag = self.output_variance(tf.reshape(tf.unique(m)[0], (-1, 1)))
        x = tf.cast(x[:, 0], tf.int32)
        patients_idx = tf.gather(self.m_pidx, x)
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
        self.beta1.prior = tfp.distributions.Gamma(to_default_float(5.0), to_default_float(3.0))
        self.beta1_raw = Parameter(to_default_float(3.0) * tf.ones(num_patients, dtype=tf.float64),
                                   transform=positive())
        self.beta1_raw.prior = tfp.distributions.Gamma(to_default_float(5.0), to_default_float(3.0))

        #self.sigma_raw = Parameter(sigma_raw, transform=positive())
        self.sigma_raw = sigma_raw

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
        #W = beta0_vec+beta1_vec * x  # Shape (Mi+Mj+..., 1)
        W = beta1_vec * x  # Shape (Mi+Mj+..., 1)

        beta0_vec = tf.reshape(tf.gather(beta0, self.m_pidx2),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...
        beta1_vec = tf.reshape(tf.gather(beta1, self.m_pidx2),
                               (-1, 1))  # Tensor of betas for each treatment of each patient. Shape equals to Mi+Mj+...
        #W2 = beta0_vec+beta1_vec * x2  # Shape (Mi+Mj+..., 1)
        W2 = beta1_vec * x2  # Shape (Mi+Mj+..., 1)

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

        #W = beta0_vec+beta1_vec * x  # Shape (Mi+Mj+..., 1)
        W = beta1_vec * x  # Shape (Mi+Mj+..., 1)
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

    def __init__(self, decay: float, sensitivity: float, variance:float, lengthscales: float, active_dims: Optional[ActiveDims]):
        super().__init__(active_dims=active_dims)

        self.d = Parameter(decay, transform=positive())
        self.s = Parameter(sensitivity, transform=positive())
        self.d.prior = tfp.distributions.Gamma(to_default_float(2.0), to_default_float(1.0))
        self.s.prior = tfp.distributions.Gamma(to_default_float(2.0), to_default_float(1.0))

        self.l = Parameter(lengthscales, transform=positive())
        #self.v = Parameter(variance, transform=positive())

    def K(self, x, x2=None):
        if x2 is None:
            x2 = x

        coeff1 = tf.math.square(self.s) * math.sqrt(math.pi) * self.l / 2
        gamma = self.d * self.l / 2
        coeff2 = tf.math.exp(tf.math.square(gamma))/(2*self.d)

        # Left part
        deltas = x - tf.transpose(x2)
        a1 = tf.math.exp(-1 * self.d * deltas)
        a2 = tf.math.exp(-1 * self.d * (x + 1))
        b1 = tf.math.erf((deltas / self.l) - gamma) + tf.transpose(tf.math.erf((x2 / self.l) + gamma))
        b2 = tf.math.erf((x / self.l) - gamma) + tf.math.erf(gamma)
        lp = a1 * b1 - a2 * b2

        # Right part
        deltas = x2 - tf.transpose(x)
        a1 = tf.math.exp(-1 * self.d * deltas)
        a2 = tf.math.exp(-1 * self.d * (x2 + 1))
        b1 = tf.math.erf((deltas / self.l) - gamma) + tf.transpose(tf.math.erf((x / self.l) + gamma))
        b2 = tf.math.erf((x2 / self.l) - gamma) + tf.math.erf(gamma)
        rp = a1 * b1 - a2 * b2

        B = coeff1 * coeff2 * (lp + tf.transpose(rp))

        # deltas = x - tf.transpose(x2)
        # B = self.v * tf.math.exp(-1*(tf.math.square(deltas)/(2.0*self.l**2)))

        figure = plt.figure()
        axes = figure.add_subplot(111)
        caxes = axes.matshow(B, interpolation='nearest')
        figure.colorbar(caxes)
        plt.show()

        return B

    def K_diag(self, x):
        coeff1 = tf.math.square(self.s) * math.sqrt(math.pi) * self.l / 2
        gamma = self.d * self.l / 2
        coeff2 = tf.math.exp(tf.math.square(gamma))/(2*self.d)

        # Left part
        deltas = x - tf.transpose(x)
        a1 = tf.math.exp(-1 * self.d * deltas)
        a2 = tf.math.exp(-1 * self.d * (x + 1))
        b1 = tf.math.erf((deltas / self.l) - gamma) + tf.transpose(tf.math.erf((x / self.l) + gamma))
        b2 = tf.math.erf((x / self.l) - gamma) + tf.math.erf(gamma)
        lp = a1 * b1 - a2 * b2

        B = coeff1 * coeff2 * (lp + tf.transpose(lp))

        # deltas = x - tf.transpose(x)
        # B = self.v * tf.math.exp(-1*(tf.math.square(deltas)/(2.0*self.l**2)))

        B = tf.linalg.diag_part(B)

        return B
