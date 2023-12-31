# Libraries
library("rstan")
options(mc.cores = as.numeric(Sys.getenv('SLURM_CPUS_PER_TASK',8)))
#rstan_options(javascript=FALSE)

# Preparation of data for .stan model
#setwd("../../../../")
task = Sys.getenv('SLURM_ARRAY_TASK_ID')
args = commandArgs(trailingOnly=TRUE)
period = args[1]

# Training data
N = read.table(paste('./data/processed_data/N.txt', sep=''), sep=' ')[,c(1)]
M = read.table(paste('./data/processed_data/M.txt', sep=''), sep=' ')[,c(1)]
y = read.table(paste('./data/processed_data/y.txt', sep=''), sep=' ')
t = read.table(paste('./data/processed_data/t.txt', sep=''), sep=' ')
x1 = read.table(paste('./data/processed_data/x1.txt', sep=''), sep=' ')[,c(1)]
x2 = read.table(paste('./data/processed_data/x2.txt', sep=''), sep=' ')[,c(1)]
x = read.table(paste('./data/processed_data/x.txt', sep=''), sep=' ')[,c(1)]
tx = read.table(paste('./data/processed_data/tx.txt', sep=''), sep=' ')[,c(1)]
hypers = read.table(paste('./data/processed_data/params.txt', sep=''), sep=' ')
trend_p = read.table(paste('./data/processed_data/trend_p.txt', sep=''), sep=' ')[,c(1)]
P = hypers[1,1]
N_max = hypers[2,1]
M_max = hypers[3,1]
T_max = hypers[4,1]
PM = hypers[5,1]
Mcumsum = cumsum(M)
Mcumsum = c(0, Mcumsum)

its = 4000

data = list(P=P,N=N,M=M,N_max=N_max,M_max=M_max,T_max=T_max,PM=PM,y=y,t=t,x1=x1,x2=x2,tx=tx, Mcumsum=Mcumsum, trend_p=trend_p)


# Run model
fit_obj = stan(file = "./src/models/parametric/PResp/SimpleModelHier.stan",
               data = data, iter = its, warmup=its/2, chains = 2, cores=8,
               init = list(list(beta1 = 0.07, beta1_p = rep(0.07,P), beta2 = 0.07, beta2_p = rep(0.07,P), tx_star = tx, alpha1 = 0.35, alpha1_p = rep(0.35,P), alpha2 = 0.433, alpha2_p = rep(0.433,P), sig_y=rep(0.58,P), sig_t= 0.516 ),
                           list(beta1 = 0.08, beta1_p = rep(0.08,P), beta2 = 0.07, beta2_p = rep(0.07,P), tx_star = tx+0.033, alpha1 = 0.333, alpha1_p = rep(0.333,P), alpha2 = 0.416, alpha2_p = rep(0.416,P), sig_y=rep(0.6,P), sig_t= 0.566 )),
               pars = c('resp_sum', 'resp_sum1','resp_sum2','log_lik','alpha1_p','alpha2_p','beta1_p','beta2_p', 'tx_star', 'sig_y'), include=TRUE, save_warmup=FALSE)


# Save results
samples = extract(fit_obj, permuted = TRUE) 

# Loo computations
loo1 = loo(fit_obj, pars = "log_lik")
print(loo1)
png(paste('./data/results_data/parametric/PResp/loo_',task,'.png', sep=''))
plot(loo1)
dev.off()

# Fitting statistics
# samples_summary = summary(fit_obj)
# print(samples_summary$summary[c('log_lik_sum_train'),])
# print(samples_summary$summary[c('log_lik_sum_test'),])

# Making plots
# pdf('./results_data/hist_alpha.pdf')
# plot(fit_obj, show_density = TRUE, pars = c("alpha"), ci_level = 0.95, fill_color = "purple")
# dev.off()
# 
# pdf('./results_data/hist_beta1.pdf')
# plot(fit_obj, show_density = TRUE, pars = c("beta1"), ci_level = 0.95, fill_color = "blue")
# dev.off()
# 
# pdf('./results_data/hist_beta2.pdf')
# plot(fit_obj, show_density = TRUE, pars = c("beta2"), ci_level = 0.95, fill_color = "blue")
# dev.off()
# 
# pdf('./results_data/hist_sig_t.pdf')
# plot(fit_obj, show_density = TRUE, pars = c("sig_t"), ci_level = 0.95, fill_color = "blue4")
# dev.off()
# 
png(paste('./data/results_data/parametric/PResp/hist_alpha1_p_',task,'.png', sep=''))
plot(fit_obj, show_density = TRUE, pars = c("alpha1_p"), ci_level = 0.95, fill_color = "purple")
dev.off()

png(paste('./data/results_data/parametric/PResp/hist_alpha2_p_',task,'.png', sep=''))
plot(fit_obj, show_density = TRUE, pars = c("alpha2_p"), ci_level = 0.95, fill_color = "purple")
dev.off()

png(paste('./data/results_data/parametric/PResp/hist_beta1_p_',task,'.png', sep=''))
plot(fit_obj, show_density = TRUE, pars = c("beta1_p"), ci_level = 0.95, fill_color = "blue")
dev.off()

png(paste('./data/results_data/parametric/PResp/hist_beta2_p_',task,'.png', sep=''))
plot(fit_obj, show_density = TRUE, pars = c("beta2_p"), ci_level = 0.95, fill_color = "blue")
dev.off()


# Time corrections
samples_y = data.frame(matrix(0.0,nrow = P,ncol = M_max))
for (p in 1:P) {
  for (m in 1:M[p]) {
    samples_y[p,m] = mean(samples$tx_star[,Mcumsum[p]+m])
  }
}
write.csv(samples_y,paste('./data/results_data/parametric/PResp/time_corrections.csv', sep=''), row.names = FALSE)

samples_y = data.frame(matrix(0.0,nrow = P,ncol = M_max))
for (p in 1:P) {
  for (m in 1:M[p]) {
    samples_y[p,m] = tx[Mcumsum[p]+m]
  }
}
write.csv(samples_y,paste('./data/results_data/parametric/PResp/true_times.csv', sep=''), row.names = FALSE)



# 
# pdf('./results_data/hist_sig_y.pdf')
# plot(fit_obj, show_density = TRUE, pars = c("sig_y"), ci_level = 0.95, fill_color = "cornflowerblue")
# dev.off()
# 
# pdf('./results_data/trace_alpha.pdf')
# traceplot(fit_obj, pars = "alpha", inc_warmup=FALSE)
# dev.off()
# 
# pdf('./results_data/trace_beta1.pdf')
# traceplot(fit_obj, pars = "beta1", inc_warmup=FALSE)
# dev.off()
# 
# pdf('./results_data/trace_beta2.pdf')
# traceplot(fit_obj, pars = "beta2", inc_warmup=FALSE)
# dev.off()
# 
# pdf('./results_data/trace_sig_t.pdf')
# traceplot(fit_obj, pars = "sig_t", inc_warmup=FALSE)
# dev.off()

# Convert 3D arrays of predicted ys to 2D for 1st meal/2nd meal/overall
# Train
samples_y = data.frame(matrix(0.0,nrow = P,ncol = N_max))
for (p in 1:P) {
  for (n in 1:N[p]) {
    samples_y[p,n] = mean(samples$resp_sum[,p,n])
  }
}
write.csv(samples_y,paste('./data/results_data/parametric/PResp/samples_y_',task,'.csv', sep=''), row.names = FALSE)

samples_y = data.frame(matrix(0.0,nrow = P,ncol = N_max))
for (p in 1:P) {
  for (n in 1:N[p]) {
    samples_y[p,n] = mean(samples$resp_sum1[,p,n])
  }
}
write.csv(samples_y,paste('./data/results_data/parametric/PResp/samples_y1_',task,'.csv', sep=''), row.names = FALSE)

samples_y = data.frame(matrix(0.0,nrow = P,ncol = N_max))
for (p in 1:P) {
  for (n in 1:N[p]) {
    samples_y[p,n] = mean(samples$resp_sum2[,p,n])
  }
}
write.csv(samples_y,paste('./data/results_data/parametric/PResp/samples_y2_',task,'.csv', sep=''), row.names = FALSE)

# Test
# samples_y_test = data.frame(matrix(0.0,nrow = P_test,ncol = N_max_test))
# for (p in 1:P_test) {
#   for (n in 1:N_test[p]) {
#     samples_y_test[p,n] = mean(samples$resp_sum[,p,n])
#   }
# }
# write.csv(samples_y_test,'./results_data/parametric/PResp/samples_y_test.csv', row.names = FALSE)
# 
# samples_y_test = data.frame(matrix(0.0,nrow = P_test,ncol = N_max_test))
# for (p in 1:P_test) {
#   for (n in 1:N_test[p]) {
#     samples_y_test[p,n] = mean(samples$resp_sum1[,p,n])
#   }
# }
# write.csv(samples_y_test,'./results_data/parametric/PResp/samples_y1_test.csv', row.names = FALSE)
# 
# samples_y_test = data.frame(matrix(0.0,nrow = P_test,ncol = N_max_test))
# for (p in 1:P_test) {
#   for (n in 1:N_test[p]) {
#     samples_y_test[p,n] = mean(samples$resp_sum2[,p,n])
#   }
# }
# write.csv(samples_y_test,'./results_data/parametric/PResp/samples_y2_test.csv', row.names = FALSE)

## Save fitted params for each sample
fitted_params_patients = data.frame(matrix(NA, nrow = P, ncol = 5))

samples_y = numeric(P)
for (p in 1:P) {
  samples_y[p] = median(samples$alpha1_p[,p])
}
fitted_params_patients[1]=samples_y

samples_y = numeric(P)
for (p in 1:P) {
  samples_y[p] = median(samples$alpha2_p[,p])
}
fitted_params_patients[2]=samples_y

samples_y = numeric(P)
for (p in 1:P) {
  samples_y[p] = median(samples$beta1_p[,p])
}
fitted_params_patients[3]=samples_y

samples_y = numeric(P)
for (p in 1:P) {
  samples_y[p] = median(samples$beta2_p[,p])
}
fitted_params_patients[4]=samples_y

samples_y = numeric(P)
for (p in 1:P) {
  samples_y[p] = median(samples$sig_y[,p])
}
fitted_params_patients[5]=samples_y

write.csv(fitted_params_patients,paste('./data/results_data/parametric/PResp/fitted_params_patients_',task,'.csv', sep=''), row.names = FALSE)
