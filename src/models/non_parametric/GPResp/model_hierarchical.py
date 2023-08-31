import gpflow as gpf
import tensorflow as tf
from src.models.non_parametric.GPResp.kernels import HierarchicalCoregion, LFMKernel, HierarchicalCoregion1
import numpy as np
from gpflow.logdensities import multivariate_normal
from gpflow.models.util import data_input_to_tensor
from src.plot.non_parametric.GPResp.plot_gp_kernels import plot_treatment_kernel, plot_baseline_kernel, plot_baseline_kernel_train, plot_total_kernel_whole_train, plot_total_kernel_whole_test


class HierarchicalModel(gpf.models.GPR):
    """Class for hierarchical GP model.

    Parameters:
    data (tuple): contains list x of np.arrays with glucose times, list y of np.arrays with glucose measurements, list meals with np.arrays of meals time and values
    T (int): time of treatment effect
    baseline_kernels (list): list of gpflow baseline kernels
    treatment_base_kernel (gpf.kernels): kernel for treatment baseline
    mean_functions (list): list of gpf.mean_functions
    noise_variance (float): model noise for likelihood
    separating_interval (int): the value for separation
    train_noise (boolean): if to include train noise
   """

    def __init__(self, data, T, baseline_kernels, treatment_base_kernels, mean_functions, noise_variance=1.0,
                 separating_interval=200.0, train_noise=True):

        x, y, meals = data
        super().__init__((x[0], y[0]), None, None, noise_variance)
        self.data = data_input_to_tensor(data)
        self.N = len(x)
        self.T = T
        self.baseline_kernels = baseline_kernels
        self.mean_functions = mean_functions
        self.separating_interval = separating_interval
        meal_lengths = [tf.shape(meal)[0] for meal in meals]
        self.meals_patient_idx = tf.repeat(tf.range(self.N),
                                           meal_lengths)  # Tensor of patient's indices of shape Mi+Mj+...

        self.likelihood = [gpf.likelihoods.Gaussian(noise_variance) for _ in range(self.N)]
        for lik in self.likelihood:
            gpf.utilities.set_trainable(lik, train_noise)

        self.x_sep, self.meals_sep = self.prepare_treatment_input(x, meals)
        self.fitting_data = self.create_fitting_data(self.x_sep, tf.reshape(self.meals_sep[:, 0], (-1, 1)),
                                                     self.meals_sep[:, 1:])

        self.treatment_kernel_cor_meal1 = HierarchicalCoregion(output_dim=tf.shape(self.meals_sep)[0], rank=1,
                                                               num_patients=self.N,
                                                               meals_patient_idx=self.meals_patient_idx,
                                                               beta0=1.0, beta1=3.0, sigma_raw=0.1,
                                                               active_dims=[1, 2])

        self.treatment_kernel_cor_meal2 = HierarchicalCoregion1(output_dim=tf.shape(self.meals_sep)[0], rank=1,
                                                                num_patients=self.N,
                                                                meals_patient_idx=self.meals_patient_idx,
                                                                beta0=1.0, beta1=3.0, sigma_raw=1.0,
                                                                active_dims=[1, 3])

        self.treatment_kernel_cor_meal1.kappa.assign(np.ones(self.treatment_kernel_cor_meal1.kappa.shape) * 1e-12)
        gpf.set_trainable(self.treatment_kernel_cor_meal1.kappa, False)
        gpf.set_trainable(self.treatment_kernel_cor_meal1.beta0, False)
        gpf.set_trainable(self.treatment_kernel_cor_meal1.beta0_raw, False)

        self.treatment_kernel_cor_meal2.kappa.assign(np.ones(self.treatment_kernel_cor_meal2.kappa.shape) * 1e-12)
        gpf.set_trainable(self.treatment_kernel_cor_meal2.kappa, False)
        gpf.set_trainable(self.treatment_kernel_cor_meal2.beta0, False)
        gpf.set_trainable(self.treatment_kernel_cor_meal2.beta0_raw, False)

        self.treatment1_base_kernel = treatment_base_kernels[0]
        self.treatment2_base_kernel = treatment_base_kernels[1]
        self.treatment_kernel = self.treatment1_base_kernel * self.treatment_kernel_cor_meal1 + self.treatment2_base_kernel * self.treatment_kernel_cor_meal2
        self.block_diag_scatter_ids_train = self.get_block_diag_scatter_ids(x)

    def prepare_treatment_input(self, x, m):
        """Prepares tensors of glucose and meals data, using separating intervals.

        Parameters:
        x (list): list x of np.arrays with glucose times of shape (Ni,1)
        m (list): list meals with np.arrays of meals time and 3 values of shape (M,4)

        Returns:
        x_sep (tf.Tensor): concatenated tensor of glucose times for all patients of shape (Ni*P,1)
        m_sep (tf.Tensor): concatenated tensor of meals times and values for all patients (Mi*P,2)
       """
        x_lengths = [tf.shape(xi)[0] for xi in x]
        x_flat = tf.cast(tf.concat(x, axis=0), tf.float64)
        intervals = tf.cast(tf.stack([n * self.separating_interval for n in range(self.N)]), tf.float64)
        x_sep = x_flat + tf.reshape(tf.repeat(intervals, x_lengths),
                                    (-1, 1))  # Concatenate glucose times for all patients

        if m is not None:
            m_lengths = [tf.shape(mi)[0] for mi in m]
            mtimes_flat = tf.reshape(tf.concat([mi[:, 0] for mi in m], axis=0) + tf.repeat(intervals,
                                                                                           m_lengths),
                                     (-1, 1))  # Concatenated times of meals for all the patients
            mvals_flat = tf.concat([mi[:, 1:] for mi in m], axis=0)  # Concatenated values of meals for all the patients
            m_sep = tf.concat([mtimes_flat, mvals_flat],
                              axis=1)  # 1st col=concat meal times for all patients, 2nd/3d/4th col=concat meals vals for all patients
        else:
            m_sep = []

        return x_sep, m_sep

    def create_fitting_data(self, x, meals_times, meals_vals, x_new=None, meals_times_new=None, meals_vals_new=None):
        """Prepares tensor of times deltas between glucose measurements and meals, their indices.

        Parameters:
        x (tf.Tensor): concatenated tensor of glucose times for all patients of shape (Ni*P,1)
        meals_times (tf.Tensor): concatenated tensor of meals times for all patients (Mi*P,1)
        meals_vals (tf.Tensor): concatenated tensor of 2 meals vals for all patients (Mi*P,2)

        Returns:
        deltas (tf.Tensor): concatenated tensor of glucose times deltas, meals values, meals indices for all patients of shape (Ni*P,4)
        scatter_ids_2d (tf.Tensor): tensor of shape = (non-zero ids for glucose measurements for all meals squared,2)
        scatter_ids_1d (tf.Tensor): tensor of non-zero ids for glucose measurements for all meals
       """
        delta_times = x - tf.transpose(
            meals_times)  # Time difference between meals times and glucose times for all patients
        mask = tf.math.logical_and(delta_times > 0.0, delta_times <= self.T)  # Mask for good glucose measurements
        times_coord = tf.experimental.numpy.nonzero(tf.transpose(mask))[
            1]  # Coordinates of good glucose measurements for every meal

        if x_new is None or meals_times_new is None:
            delta_times_new = delta_times
            times_coord_new = times_coord
            meals_vals_new = meals_vals
            mask_new = mask
        else:
            delta_times_new = x_new - tf.transpose(meals_times_new)
            mask_new = tf.math.logical_and(delta_times_new > 0.0, delta_times_new <= self.T)
            times_coord_new = tf.experimental.numpy.nonzero(tf.transpose(mask_new))[1]
        delta_times_suitable_new = tf.transpose(delta_times_new)[tf.transpose(mask_new)]

        scatter_ids_1d = tf.reshape(times_coord_new,
                                    (-1, 1))  # Tensor of non-zero ids for glucose measurements for all meals

        xx, yy = tf.meshgrid(times_coord_new, times_coord)
        scatter_ids_2d = tf.stack([tf.reshape(yy, [-1]), tf.reshape(xx, [-1])],
                                  axis=1)  # Tensor of shape = (non-zero ids for glucose measurements for all meals) squared,2)

        num_obs_new = tf.reduce_sum(tf.cast(mask_new, dtype=tf.int64),
                                    axis=0)  # Number of suitable obs in each treat region

        if 0 in num_obs_new:
            raise Exception("Sorry, there is a meal with 0 suitable glucose values.")

        delta_times_suitable_new_idx = tf.repeat(tf.range(tf.shape(num_obs_new)[0], dtype=tf.float64),
                                                 num_obs_new)  # Treatment index for each time delta value

        meals_vals_new += tf.random.normal(tf.shape(meals_vals_new), mean=0.0, stddev=0.001,
                                           dtype=tf.float64)  # Adding perturbations to meals
        meals_vals_new_pert = tf.repeat(meals_vals_new, num_obs_new,
                                        axis=0)  # Repeating the meals perturbations for all suiltable glucose measurements
        deltas = tf.concat(
            [tf.reshape(delta_times_suitable_new, (-1, 1)), tf.reshape(delta_times_suitable_new_idx, (-1, 1)),
             meals_vals_new_pert], axis=1)

        return deltas, scatter_ids_2d, scatter_ids_1d

    @staticmethod
    def get_block_diag_scatter_ids(x, x2=None):
        """Prepares tensor of glucose indices for all patients.

        Parameters:
        x (list): list of tensors of glucose times for all patients of shape (Ni,1)
        x2 (tf.Tensor): list of tensors of glucose times for all patients of shape (Ni,1)

        Returns:
        indices (tf.Tensor): concatenated tensor of glucose times indices for all patients of shape (Ni1*Ni2+Nj1*Nj2+...,2)
       """
        if x2 is None:
            x2 = x

        offset1 = 0
        offset2 = 0
        indices = []
        N = len(x)

        for n in range(N):
            N1 = tf.shape(x[n])[0]
            N2 = tf.shape(x2[n])[0]
            xx, yy = tf.meshgrid(tf.range(N2), tf.range(N1))
            indices_i = tf.transpose(
                tf.stack([tf.reshape(yy, (-1,)) + offset1, tf.reshape(xx, (-1,)) + offset2]))  # Indices array=(N1*N2,2)
            offset1 += N1
            offset2 += N2
            indices.append(indices_i)

        indices = tf.concat(indices, axis=0)  # Concat arrays under each other = (Ni1*Ni2+Nj1*Nj2+...,2)
        return indices

    def log_marginal_likelihood(self) -> tf.Tensor:
        """ Computes log-likelihood.

        Returns:
        tf.reduce_sum(log_prob) (float): log-likelihood value
        """
        x, y, meals = self.data
        deltas, scatter_ids_2d, _ = self.fitting_data

        # Calculate the sum of baseline and treatment
        K = self.K_total(x, deltas, baseline_scatter_ids=self.block_diag_scatter_ids_train,
                         treatment_scatter_ids=scatter_ids_2d, action='total')

        # Add noise to kernel
        K = self.add_noise_cov(K)
        K += 1e-9 * tf.eye(K.shape[0], dtype=K.dtype)
        L = tf.linalg.cholesky(K)

        # Calculate log-lik as sum of log probabilities for each patient
        m = tf.concat([mi(xi) for mi, xi in zip(self.mean_functions, x)],
                      axis=0)  # Calculates mean for glucose obs. of every patient
        m = tf.cast(m, tf.float64)
        ys = tf.cast(tf.concat(y, axis=0), tf.float64)
        log_prob = multivariate_normal(ys, m, L)

        return tf.reduce_sum(log_prob)

    def add_noise_cov(self, K: tf.Tensor) -> tf.Tensor:
        """ Adds noise to diagonal elements of kernel.

        Parameters:
        K (tf.Tensor): kernel

        Returns:
        tf.linalg.set_diag(K, k_diag + s_diag) (tf.Tensor): kernel with added noise to diagonal
        """
        x, _, _ = self.data
        x_lengths = [tf.shape(xi)[0] for xi in x]
        noise_variances = tf.stack([lik.variance for lik in self.likelihood])
        s_diag = tf.repeat(noise_variances,
                           x_lengths)  # Add different noise variances to glucose obseravations of dfferent patients.
        k_diag = tf.linalg.diag_part(K)

        return tf.linalg.set_diag(K, k_diag + s_diag)

    def K_total(self, x, deltas, baseline_scatter_ids, treatment_scatter_ids, x2=None, deltas2=None, action='total', time='train'):
        """ Computes total kernel.

        Parameters:
        x (list): list x of np.arrays with glucose times of shape (Ni,1)
        deltas (tf.Tensor): concatenated tensor of glucose times deltas, meals values, meals indices for all patients of shape (Ni*P,3)
        baseline_scatter_ids (tf.Tensor): concatenated tensor of glucose times indices for all patients of shape (Ni1*Ni2+Nj1*Nj2+...,2)
        treatment_scatter_ids (tf.Tensor): tensor of shape = (non-zero ids for glucose measurements for all meals) squared,2)

        Returns:
        Ksum (tf.Tensor): sum of baseline and meals kernels
        """

        if x2 is None:
            x2 = x
        if deltas2 is None:
            deltas2 = deltas

        # Calculate baseline kernel
        Ksum = self.K_base(x, scatter_ids=baseline_scatter_ids, x2=x2)
        Kb = self.K_base(x, scatter_ids=baseline_scatter_ids, x2=x2)
        if time == 'test':
            # Boolean mask for plotting baseline for exact patient 2
            patients_indices = [i for i, xi in enumerate(x)]
            patients_indices_len = [tf.shape(xi)[0] for i, xi in enumerate(x)]
            patients_idx_meal = tf.repeat(patients_indices, patients_indices_len)
            mask_boolean_meal = tf.math.equal(patients_idx_meal, 2)
            #plot_baseline_kernel_train(Ksum, mask_boolean_meal)


        if len(self.meals_sep) > 0:
            # Calculate treatment kernel
            if action == 'total':
                Kt = self.treatment_kernel(deltas, deltas2)
                Ksum = tf.tensor_scatter_nd_add(Ksum, treatment_scatter_ids, tf.reshape(Kt, [-1]))
                tlse_meal1, B_meal1 = self.treatment1_base_kernel(deltas, deltas2), self.treatment_kernel_cor_meal1(
                    deltas, deltas2)
                tlse_meal2, B_meal2 = self.treatment2_base_kernel(deltas, deltas2), self.treatment_kernel_cor_meal2(
                    deltas,
                    deltas2)

                if time == 'test':
                    # Boolean mask for plotting treatment kernel for exact patient 2
                    d_idx = tf.cast(deltas[:, 1], tf.int32)
                    patient_meals_idx_meal1 = self.treatment_kernel_cor_meal1.m_pidx
                    patients_idx_meal1 = tf.gather(patient_meals_idx_meal1, d_idx)
                    mask_boolean_meal1 = tf.math.equal(patients_idx_meal1, 2)
                    #plot_total_kernel_whole_train(B_meal1, tlse_meal1, B_meal2, tlse_meal2, Kt, Kb, Ksum, mask_boolean_meal1, mask_boolean_meal)

                return Ksum


    def K_base(self, x, scatter_ids, x2=None):
        """ Computes base kernel.

        Parameters:
        x (list): list x of np.arrays with glucose times of shape (Ni,1)
        scatter_ids (tf.Tensor): concatenated tensor of glucose times indices for all patients of shape (Ni1*Ni2+Nj1*Nj2+...,2)

        Returns:
        Kbase (tf.Tensor): baseline kernel
        """
        if x2 is None:
            x2 = x

        sparse_flattened = tf.concat([tf.reshape(kernel_base(tf.cast(x1i, tf.float64), tf.cast(x2i, tf.float64)), (-1,))
                                      for kernel_base, x1i, x2i in zip(self.baseline_kernels, x, x2)],
                                     axis=0)  # For each patient i the kernel of size (Ni,Ni) is calculated between all its glucose measurements. Then all the kernels are flattened.

        N1 = sum([tf.shape(xi)[0] for xi in x])  # Total num of glucose observations
        N2 = sum([tf.shape(xi)[0] for xi in x2])  # Total num of glucose observations
        sparse_cov = tf.sparse.SparseTensor(
            tf.cast(scatter_ids, tf.int64), sparse_flattened, [N1, N2]
        )  # If the pair in a row of scatter_ids exists, than in kernel of shape [N1, N2] there will be the element, corresponding to the index of that pair from sparse_flattened.
        Kbase = tf.sparse.to_dense(sparse_cov)

        return Kbase

    def K_base_diag(self, x, time):
        """ Computes base diagonal kernel.

        Parameters:
        x (list): list x of np.arrays with glucose times of shape (Ni,1)

        Returns:
        Kbase (tf.Tensor): baseline diagonal kernel
        """
        if time == 'test':
            # Plot baseline for kernel for test
            kernel_base_plot = self.baseline_kernels[2]
            mat = kernel_base_plot(x[2], full_cov=False)
            #plot_baseline_kernel(mat)

        return tf.concat([kernel_base(x2i, full_cov=False) for kernel_base, x2i in zip(self.baseline_kernels, x)],
                         axis=0)

    def K_treatment(self, x, d, scatter_ids, x2=None, d2=None):
        """ Computes treatment kernel.

        Parameters:
        x (tf.Tensor): concatenated tensor of glucose times for all patients of shape (Ni*P,1)
        d (tf.Tensor): concatenated tensor of glucose times deltas, meals values, meals indices for all patients of shape (Ni*P,3)
        scatter_ids (tf.Tensor):  tensor of shape = (non-zero ids for glucose measurements for all meals) squared,2)

        Returns:
        K_scatter (tf.Tensor): treatment kernel
        """
        if x2 is None:
            x2 = x
        if d2 is None:
            d2 = d

        Kt_meal1 = self.treatment1_base_kernel(d, d2) * self.treatment_kernel_cor_meal1(d, d2)
        K_scatter_meal1 = tf.matmul(tf.zeros_like(x), tf.zeros_like(x2), transpose_b=True)
        K_scatter_meal1 = tf.tensor_scatter_nd_add(K_scatter_meal1, scatter_ids, tf.reshape(Kt_meal1, [-1]))

        Kt_meal2 = self.treatment2_base_kernel(d, d2) * self.treatment_kernel_cor_meal2(d, d2)
        K_scatter_meal2 = tf.matmul(tf.zeros_like(x), tf.zeros_like(x2), transpose_b=True)
        K_scatter_meal2 = tf.tensor_scatter_nd_add(K_scatter_meal2, scatter_ids, tf.reshape(Kt_meal2, [-1]))

        return K_scatter_meal1, K_scatter_meal2

    def K_treatment_diag(self, x, d, scatter_ids, x_list, time):
        """ Computes treatment diagonal kernel (if there is no need in full covariance).

        Parameters:
        x (tf.Tensor): concatenated tensor of glucose times for all patients of shape (Ni*P,1)
        d (tf.Tensor): concatenated tensor of glucose times deltas, meals indices, carbs meals values, fat meals values for all patients of shape (Ni*P,4)
        scatter_ids (tf.Tensor):  tensor of shape = (non-zero ids for glucose measurements for all meals),1)

        Returns:
        K_scatter_diag (tf.Tensor): treatment diagonal kernel
        """
        # Boolean mask for plotting carbs and fat cov. matrix for exact patient 2
        d_idx = tf.cast(d[:, 1], tf.int32)
        patient_meals_idx_meal1 = self.treatment_kernel_cor_meal1.m_pidx
        patient_meals_idx_meal2 = self.treatment_kernel_cor_meal2.m_pidx
        patients_idx_meal1 = tf.gather(patient_meals_idx_meal1, d_idx)
        patients_idx_meal2 = tf.gather(patient_meals_idx_meal2, d_idx)
        mask_boolean_meal1 = tf.math.equal(patients_idx_meal1, 2)
        mask_boolean_meal2 = tf.math.equal(patients_idx_meal2, 2)

        Kt_meal1 = self.treatment1_base_kernel(d, full_cov=False) * self.treatment_kernel_cor_meal1(d, full_cov=False)
        K_scatter_diag_meal1 = tf.reshape(tf.zeros_like(x), (-1,))
        K_scatter_diag_meal1 = tf.tensor_scatter_nd_add(K_scatter_diag_meal1, scatter_ids, Kt_meal1)


        Kt_meal2 = self.treatment2_base_kernel(d, full_cov=False) * self.treatment_kernel_cor_meal2(d, full_cov=False)
        K_scatter_diag_meal2 = tf.reshape(tf.zeros_like(x), (-1,))
        K_scatter_diag_meal2 = tf.tensor_scatter_nd_add(K_scatter_diag_meal2, scatter_ids, Kt_meal2)

        # Plot cov. for fat
        if time == 'test':
            kernel_base_plot = self.baseline_kernels[2]
            Kb = kernel_base_plot(x_list[2], full_cov=False)
            #plot_total_kernel_whole_test(Kt_meal1, Kt_meal2, Kb, mask_boolean_meal1, mask_boolean_meal2)

        return K_scatter_diag_meal1, K_scatter_diag_meal2

    def predict_train(self, x, meals, patients_meals_idx, time='train'):
        """ Predicts means and variances for glucose data.

        Parameters:
        x (list): list x of np.arrays with glucose times of shape (Ni,1)
        meals (list): list meals with np.arrays of meals time and 3 values of shape (M,4)
        patients_meals_idx (tf.Tensor): tensor of patients indices for every meal

        Returns:
        f_mean_zero + mnew (tf.Tensor): means of predictions for meals values
        f_var (tf.Tensor): variance of predictions for meals values
        """
        f_mean_zero_meal1, f_mean_zero_meal2, f_var_meal1, f_var_meal2 = self.predict_multiple(x, meals,
                                                                                               pm_idx=patients_meals_idx,
                                                                                               full_cov=False,
                                                                                               time=time)
        mnew = tf.concat([mf(Xi) for mf, Xi in zip(self.mean_functions, x)], axis=0)

        return f_mean_zero_meal1 + mnew, f_mean_zero_meal2 + mnew, f_var_meal1, f_var_meal2

    def predict_multiple(self, x, meals, pm_idx, full_cov: bool = False, time='train'):
        """ Calculates predictions for all glucose data, passed at once.

        Parameters:
        x (list): list x of np.arrays with glucose times of shape (Ni,1)
        meals (list): list meals with np.arrays of meals time and 3 values of shape (M,4)
        pm_idx (tf.Tensor): tensor of patients indices for every meal
        full_cov (boolean): if return full covariance or diagonal

        Returns:
        mnew (tf.Tensor): means of predictions for meals values
        f_var (tf.Tensor): variance of predictions for meals values
        """
        x_sep_new, meals_sep_new = self.prepare_treatment_input(x, meals)
        if len(meals_sep_new) == 0 or len(self.meals_sep) == 0:
            return tf.zeros_like(x_sep_new), tf.zeros_like(meals_sep_new)

        # Kmm kernel
        X, Y, M = self.data
        deltas, scatter_ids_2d, _ = self.fitting_data
        self.treatment_kernel_cor_meal1.set_meals_pidx(self.meals_patient_idx)
        self.treatment_kernel_cor_meal1.set_meals_pidx2(self.meals_patient_idx)
        self.treatment_kernel_cor_meal2.set_meals_pidx(self.meals_patient_idx)
        self.treatment_kernel_cor_meal2.set_meals_pidx2(self.meals_patient_idx)
        kmm = self.K_total(X, deltas, baseline_scatter_ids=self.block_diag_scatter_ids_train,
                           treatment_scatter_ids=scatter_ids_2d, action='total', time=time)
        kmm_noisy = self.add_noise_cov(kmm)

        # Kmn kernel
        deltas_new, scatter_ids_2d_new, scatter_ids_1d_new = self.create_fitting_data(self.x_sep,
                                                                                      tf.reshape(self.meals_sep[:, 0],
                                                                                                 (-1, 1)),
                                                                                      self.meals_sep[:, 1:],
                                                                                      x_new=x_sep_new,
                                                                                      meals_times_new=tf.reshape(
                                                                                          meals_sep_new[:, 0], (-1, 1)),
                                                                                      meals_vals_new=meals_sep_new[:,
                                                                                                     1:])
        self.treatment_kernel_cor_meal1.set_meals_pidx(self.meals_patient_idx)
        self.treatment_kernel_cor_meal1.set_meals_pidx2(pm_idx)
        self.treatment_kernel_cor_meal2.set_meals_pidx(self.meals_patient_idx)
        self.treatment_kernel_cor_meal2.set_meals_pidx2(pm_idx)
        kmn_meal1, kmn_meal2 = self.K_treatment(self.x_sep, deltas, scatter_ids_2d_new, x_sep_new, deltas_new)

        # Knn kernel
        self.treatment_kernel_cor_meal1.set_meals_pidx(pm_idx)
        self.treatment_kernel_cor_meal1.set_meals_pidx2(pm_idx)
        self.treatment_kernel_cor_meal2.set_meals_pidx(pm_idx)
        self.treatment_kernel_cor_meal2.set_meals_pidx2(pm_idx)

        if full_cov:
            _, scatter_ids_2d_new, _ = self.create_fitting_data(x_sep_new, tf.reshape(meals_sep_new[:, 0], (-1, 1)),
                                                                meals_sep_new[:, 1:])
            knn_meal1, knn_meal2 = self.K_treatment(x_sep_new, deltas_new, scatter_ids_2d_new)
        else:
            knn_meal1, knn_meal2 = self.K_treatment_diag(x_sep_new, deltas_new, scatter_ids_1d_new, x, time)

        glucose_means = tf.cast(tf.concat([mf(Xi) for mf, Xi in zip(self.mean_functions, X)], axis=0), tf.float64)
        ys = tf.cast(tf.concat(Y, axis=0), tf.float64)
        err = ys - glucose_means
        conditional = gpf.conditionals.base_conditional
        f_mean_zero_meal1, f_var_meal1 = conditional(
            kmn_meal1, kmm_noisy, knn_meal1, err, full_cov=full_cov, white=False
        )
        f_mean_zero_meal2, f_var_meal2 = conditional(
            kmn_meal2, kmm_noisy, knn_meal2, err, full_cov=full_cov, white=False
        )

        return f_mean_zero_meal1, f_mean_zero_meal2, f_var_meal1, f_var_meal2

    def predict_baseline(self, x, full_cov: bool = False, time='train'):
        """ Calculates predictions for all glucose data, passed at once.

        Parameters:
        x (list): list x of np.arrays with glucose times of shape (Ni,1)
        full_cov (boolean): if return full covariance or diagonal

        Returns:
        mnew (tf.Tensor): means of predictions for glucose values
        f_var (tf.Tensor): variance of predictions for glucose values
        """

        # Kmm kernel
        X, Y, M = self.data
        deltas, scatter_ids_2d, _ = self.fitting_data
        self.treatment_kernel_cor_meal1.set_meals_pidx(self.meals_patient_idx)
        self.treatment_kernel_cor_meal1.set_meals_pidx2(self.meals_patient_idx)
        self.treatment_kernel_cor_meal2.set_meals_pidx(self.meals_patient_idx)
        self.treatment_kernel_cor_meal2.set_meals_pidx2(self.meals_patient_idx)
        kmm = self.K_total(X, deltas, baseline_scatter_ids=self.block_diag_scatter_ids_train,
                           treatment_scatter_ids=scatter_ids_2d, action='total')
        kmm_noisy = self.add_noise_cov(kmm)

        # Kmn kernel
        baseline_scatter_ids = self.get_block_diag_scatter_ids(X, x)
        kmn = self.K_base(X, scatter_ids=baseline_scatter_ids, x2=x)

        # Knn kernel
        if full_cov:
            baseline_scatter_ids = self.get_block_diag_scatter_ids(x)
            knn = self.K_base(x, scatter_ids=baseline_scatter_ids)
        else:
            knn = self.K_base_diag(x, time)

        glucose_means = tf.concat([mf(Xi) for mf, Xi in zip(self.mean_functions, X)], axis=0)
        ys = tf.cast(tf.concat(Y, axis=0), tf.float64)
        err = ys - glucose_means
        conditional = gpf.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm_noisy, knn, err, full_cov=full_cov, white=False
        )
        mnew = tf.concat([mf(xi) for mf, xi in zip(self.mean_functions, x)], axis=0)
        f_mean = f_mean_zero + mnew

        return f_mean, f_var
