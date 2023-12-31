data { 
int<lower=1> P; // number of individuals  
int<lower=0> N[P]; // number of glucose observations for each patient
int<lower=0> M[P]; // number of meals for each patient
int<lower=1> N_max; // max. number of glucose observations across all patients
int<lower=1> M_max; // max. number of meals observations across all patients
real<lower=1.0> T_max; // max. time of glucose observations across all patients
int<lower=1> PM; // number of patients, multiplied by number of meals
matrix[P, N_max] y;  // glucose observations for each patient
matrix[P, N_max] t;  // timestamps of glucose observations for each patient
vector[PM] x1; // carbs meals' values for each patient for each meal
vector[PM] x2; // fat meals' values for each patient for each meal
vector[PM] tx; // times of meals' values for each patient for each meal
int<lower=0> Mcumsum[P+1]; // cumulative sum of number of meals for each patient
real<lower=0> trend_p[P]; // median trend for all the patients
}   

parameters { 
vector[PM] tx_star; // true time of meals  
real<lower=0> sig_t; // std of the time uncertainty  
real sig_y[P]; // std of the patients' likelihood  
real beta1; // hyper-prior for betas for carbs
real beta1_p[P]; // carbs betas for each patient
real beta2; // hyper-prior for betas for fat
real beta2_p[P]; // fat betas for each patient
real alpha1; // hyper-prior for alphas for carbs
real alpha1_p[P];  // alphas for each patient for carbs
real coeff_alpha2; // hyper-prior for alphas coeff for fat
real coeff_alpha2_p[P];  // alphas for each patient coeff for fat

} 

model {  
  
real meals[M_max];  // array to store sampled meals value for each patient
meals = rep_array(0, M_max); 

// Priors for the parameters
sig_t ~ normal(0.166,0.083);
tx_star ~ normal(tx, sig_t);
beta1 ~ normal(0.15,0.1);
beta2 ~ normal(0.15,0.1);
alpha1 ~ normal(0.333, 0.166);
coeff_alpha2 ~ normal(1.15, 0.2);


for (p in 1:P){ 
  // Priors for the parameters
  sig_y[p] ~ normal(0,0.5);  
  alpha1_p[p] ~ normal(alpha1,0.166);
  coeff_alpha2_p[p] ~ normal(coeff_alpha2,0.1);
  beta1_p[p] ~ normal(beta1,0.05); 
  beta2_p[p] ~ normal(beta2,0.05);
  
  for (n in 1:N[p]){  

    for (m in 1:M[p]) { 
      if (fabs(tx[Mcumsum[p]+m]-t[p,n])<4){
        
        meals[m] = ((x1[Mcumsum[p]+m]*beta1_p[p])*exp(-0.5*(t[p,n]-tx_star[Mcumsum[p]+m]-3*(alpha1_p[p]))^2/(alpha1_p[p])^2))  +  ((x2[Mcumsum[p]+m]*beta2_p[p])*exp(-0.5*(t[p,n]-tx_star[Mcumsum[p]+m]-3*(alpha1_p[p]*coeff_alpha2_p[p]))^2/(alpha1_p[p]*coeff_alpha2_p[p])^2)); 
      
      }
    }  
    y[p,n] ~ normal(sum(meals)+trend_p[p], sig_y[p]); 
    
    }
    
  } 
}


generated quantities{
// Train
real resp[P, N_max, M_max];
matrix[P, N_max] resp_sum;

real resp1[P, N_max, M_max];
matrix[P, N_max] resp_sum1;

real resp2[P, N_max, M_max];
matrix[P, N_max] resp_sum2;

matrix[P, N_max] log_lik_train;
row_vector[N_max] log_lik;

// Train
resp = rep_array(0, P, N_max, M_max);
resp_sum = rep_matrix(0, P, N_max);

resp1 = rep_array(0, P, N_max, M_max);
resp_sum1 = rep_matrix(0, P, N_max);

resp2 = rep_array(0, P, N_max, M_max);
resp_sum2 = rep_matrix(0, P, N_max); 

log_lik_train = rep_matrix(0, P, N_max);
log_lik= rep_row_vector(0, N_max);

// Train
for (p in 1:P) {
  for (n in 1:N[p]) {
    for (m in 1:M[p]) {
      if (fabs(tx[Mcumsum[p]+m]-t[p,n])<4){
        
        resp1[p,n,m] = (x1[Mcumsum[p]+m]*beta1_p[p])*exp(-0.5*(t[p,n]-tx_star[Mcumsum[p]+m]-3*(alpha1_p[p]))^2/(alpha1_p[p])^2);
        resp2[p,n,m] = (x2[Mcumsum[p]+m]*beta2_p[p])*exp(-0.5*(t[p,n]-tx_star[Mcumsum[p]+m]-3*(coeff_alpha2_p[p]*alpha1_p[p]))^2/(coeff_alpha2_p[p]*alpha1_p[p])^2);
        resp[p,n,m] = resp1[p,n,m]  +  resp2[p,n,m];
        
        
      }
    }

    resp_sum[p,n] = sum(resp[p,n,])+trend_p[p];
    resp_sum1[p,n] = sum(resp1[p,n,]);
    resp_sum2[p,n] = sum(resp2[p,n,]);
    
    log_lik_train[p,n]= normal_lpdf(y[p,n] | resp_sum[p,n], sig_y[p]); 
  }
log_lik+=log_lik_train[p];
}

}
