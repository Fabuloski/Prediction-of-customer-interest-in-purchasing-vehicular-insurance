# AI534
# IA2 skeleton code
import numpy as np, pandas as pd, matplotlib.pyplot as plt, random

# Loads a data file from a provided file location.
def load_data(path):
    loaded_data = pd.read_csv(path)
    return loaded_data

# Implements dataset preprocessing. For this assignment, you just need to implement normalization 
# of the three numerical features.

def preprocess_data(data, normalize:bool=False, test:bool=False):
    preprocessed_data = data.copy()  

    # normalize data if option to normalize selected
    if normalize:
        global params
        params = {}
        for col in preprocessed_data.columns:
            if col == "Age" or col== "Annual_Premium" or col == "Vintage":
                μ = np.mean(preprocessed_data[col])
                σ = np.std(preprocessed_data[col])
                preprocessed_data[col]   = (preprocessed_data[col] -μ)/ σ
                params[col] = (μ, σ)
    if test:
         for col in preprocessed_data.columns:
            if col == "Age" or col== "Annual_Premium" or col == "Vintage":
                μ = params[col][0]
                σ = params[col][1]
                preprocessed_data[col]   = (preprocessed_data[col] -μ)/ σ
    
    # return preprocessed data   
    return preprocessed_data

# Trains a logistic regression model with L2 regularization on the provided train_data, using the supplied lambd
# weights should store the per-feature weights of the learned logisitic regression model. train_acc and val_acc 
# should store the training and validation accuracy respectively. 
def LR_L2_train(train_data, val_data, λ, α, ϵ ,n_iter=4000):
    x_train = np.transpose(train_data.drop("Response", axis=1).to_numpy())   
    n_train = train_data.shape[0]
    n_features = train_data.shape[1] - 1
    y_train = train_data["Response"].to_numpy()
    
    x_val = np.transpose(val_data.drop("Response", axis=1).to_numpy())
    y_val = val_data["Response"].to_numpy()
    n_val = val_data.shape[0]
      
    w = np.ones(n_features) 
    convergence =np.zeros(n_iter + 1)
       
    for j in range(n_iter): 
        loss_train = 0
        loss_val = 0
        Bₙ_train = np.zeros(n_features)
      
        con_train = 0 
         
        ##Training Set 
        for i in range(n_train):
            yᵢ_train = y_train[i]
            prob_train = 1 / (1 + np.exp(-1 * np.dot(w, x_train[:, i])))
            if prob_train>= 0.5:
                ŷᵢ_train = 1
            else:
                ŷᵢ_train = 0 
                
            rem_train = yᵢ_train - prob_train
            Bₙ_train  += np.dot(rem_train, x_train[:, i])
            
            if yᵢ_train==ŷᵢ_train:
                loss_train += 1
            con_train += -yᵢ_train * np.log(prob_train) - (1 - yᵢ_train) * np.log(1 - prob_train)
            
        convergence[j + 1] = con_train / n_train + λ * np.sum(np.square(w[1 : ]))
        diff = convergence[j + 1] - convergence[j]
        train_acc = loss_train / n_train
        
        if abs(diff)>ϵ:
            w += (α / n_train) * Bₙ_train
            for k in range(1, n_features - 1):
                w[k] -= α * λ * w[k]
        else:
            break
    
    for i in range(n_val):
        yᵢ_val = y_val[i]
        prob_val = 1 / (1 + np.exp(-1 * np.dot(w, x_val[:, i])))
        if prob_val>= 0.5:
            ŷᵢ_val = 1
        else:
            ŷᵢ_val = 0 
        if yᵢ_val==ŷᵢ_val:
            loss_val += 1
            
    val_acc = loss_val / n_val

    return w, train_acc, val_acc

# Trains a logistic regression model with L1 regularization on the provided train_data, using the supplied lambd
# weights should store the per-feature weights of the learned logisitic regression model. train_acc and val_acc 
# should store the training and validation accuracy respectively. 
def LR_L1_train(train_data, val_data, λ, α, ϵ ,n_iter=4000):
    x_train = np.transpose(train_data.drop("Response", axis=1).to_numpy())   
    n_train = train_data.shape[0]
    n_features = train_data.shape[1] - 1
    y_train = train_data["Response"].to_numpy()
    
    x_val = np.transpose(val_data.drop("Response", axis=1).to_numpy())
    y_val = val_data["Response"].to_numpy()
    n_val = val_data.shape[0]
    
    w = np.ones(n_features) 
    convergence =np.zeros(n_iter + 1)
       
    for j in range(n_iter): 
        loss_train = 0
        loss_val = 0
        Bₙ_train = np.zeros(n_features)
      
        con_train = 0 
        
        ##Training Set 
        for i in range(n_train):
            yᵢ_train = y_train[i]
            prob_train = 1 / (1 + np.exp(-1 * np.dot(w, x_train[:, i])))
            if prob_train>= 0.5:
                ŷᵢ_train = 1
            else:
                ŷᵢ_train = 0 
                
            rem_train = yᵢ_train - prob_train
            Bₙ_train  += np.dot(rem_train, x_train[:, i])
            
            if yᵢ_train==ŷᵢ_train:
                loss_train += 1
            con_train += -yᵢ_train * np.log(prob_train) - (1 - yᵢ_train) * np.log(1 - prob_train)
            
        convergence[j + 1] = con_train / n_train + λ * np.sum(np.abs(w[1 : ]))
        diff = convergence[j + 1] - convergence[j]
        train_acc = loss_train / n_train
        
        if abs(diff)>ϵ:
            w += (α / n_train) * Bₙ_train
            for k in range(1, n_features - 1):
                w[k] = np.sign(w[k]) * (np.max([np.abs(w[k]) - α * λ, 0])) 
        else:
            break
    
    for i in range(n_val):
        yᵢ_val = y_val[i]
        prob_val = 1 / (1 + np.exp(-1 * np.dot(w, x_val[:, i])))
        if prob_val>= 0.5:
            ŷᵢ_val = 1
        else:
            ŷᵢ_val = 0 
        if yᵢ_val==ŷᵢ_val:
            loss_val += 1
            
    val_acc = loss_val / n_val

    return w, train_acc, val_acc, j

# Generates and saves plots of the accuracy curves. Note that you can interpret accs as a matrix
# containing the accuracies of runs with different lambda values and then put multiple loss curves in a single plot.
def plot_accuracy(acc, i, save, ridge=False, noise=False):
    fig, ax = plt.subplots(figsize=[16,9])
    plt.ylabel('Accuracy', fontweight="bold")
    plt.xlabel('i', fontweight="bold")

    if ridge:
        title_string = "Ridge"
    else:
        title_string = "Lasso"
    if noise:
         plt.title(f"Accuracy of model in respect to regularization parameter 10 ^ i, {title_string}; Noisy data", fontweight="bold")
    else:
        plt.title(f"Accuracy of model in respect to regularization parameter 10 ^ i, {title_string}", fontweight="bold")
        
    train_acc = [acc[i][0] for i in range(len(acc))]
    val_acc = [acc[i][1] for i in range(len(acc))]
    
    plt.plot(i, train_acc, label="Training")
    plt.plot(i, val_acc, label= "Validation")

    ax.legend(loc='upper right')
    plt.savefig(save)

    plt.show()
    
#Sparsity Plot Function
def plot_sparsity(weights, i, save):
    sparsity = np.zeros(len(i))
    for j in range(len(i)):
        sparsity[j] = np.sum(weights[j]<=10 ** -6)
    
    fig, ax = plt.subplots(figsize=[16,9])
    plt.ylabel('Sparsity', fontweight="bold")

    plt.xlabel('λ [10^i]', fontweight="bold")
    plt.title(f"Sparsity of weights in respect to λ [10 ^ i]", fontweight="bold")  
    plt.plot(i, sparsity)
   
    plt.savefig(save)

    plt.show()

# Invoke the above functions to implement the required functionality for each part of the assignment.
# Part 0  : Data preprocessing.
train_data = load_data('IA2-train.csv')
valid_data = load_data('IA2-dev.csv')
noisy_data = load_data("IA2-train-noisy.csv")

data = {
    "processed_data" : preprocess_data(train_data, normalize=True),
    "processed_val"  : preprocess_data(valid_data, test=True),
    "noisy_data"     : preprocess_data(noisy_data, normalize=True)
       }


# Part 1 . Implement logistic regression with L2 regularization and experiment with different lambdas
#Part 1A
ϵ = 10**-4
i = [i for i in range(-4, 3, 1)]
λs = [10**i for i in i]
αs = [3, 3, 1, 1, 0.1, 0.01, 10**-2]
acc = [LR_L2_train(data["processed_data"], data["processed_val"], λs[k], αs[k], ϵ)[1:3] for (k,i) in enumerate(i)]
plot_accuracy(acc, i, "Accuracy Plot", True)

#Part 1B
λs_opt = [10**-4 , 10**-3, 10**-2]
αs_opt = [3, 3, 1]
w_low, w_opt, w_high = [LR_L2_train(data["processed_data"], data["processed_val"], λ, αs_opt[k], ϵ)[0] for (k, λ) in enumerate(λs_opt)]
column = data["processed_data"].drop(["Response", "dummy"], axis = 1).columns
pd.DataFrame(np.abs(w_low[1:]), column, columns=['Weights'] ).sort_values(by= "Weights", ascending=False).head()
pd.DataFrame(np.abs(w_opt[1:]), column, columns=['Weights'] ).sort_values(by= "Weights", ascending=False).head()
pd.DataFrame(np.abs(w_high[1:]), column, columns=['Weights'] ).sort_values(by= "Weights", ascending=False).head()

#Part 1C
weights_clean = [LR_L2_train(data["processed_data"], data["processed_val"], λ, αs[k], ϵ)[0] for (k, λ) in enumerate(λs)]
plot_sparsity(weights_clean, i, "Sparsity plot")

# Part 2  Training and experimenting with IA2-train-noisy data.
i = [i for i in range(-4, 3, 1)]
λs = [10**i for i in i]
αs_noise = [2, 2, 2, 2, 10**-1, 10**-2, 10**-2]
acc_noisy = [LR_L2_train(data["noisy_data"], data["processed_val"], λs[k], αs_noise[k], ϵ)[1:3] for (k,i) in enumerate(i)]
plot_accuracy(acc_noisy, i, "Noisy data", True, True)

# Part 3  Implement logistic regression with L1 regularization and experiment with different lambdas
# Part 3A
i = [i for i in range(-4, 3, 1)]
λs = [10**i for i in i]
αs_L1 = [3, 3, 3, 3, 0.1, 0.01, 10**-5]
acc_L1 = [LR_L1_train(data["processed_data"], data["processed_val"], λs[k], αs_L1[k], ϵ)[1:3] for (k,i) in enumerate(i)]
L1_weights = [LR_L1_train(data["processed_data"], data["processed_val"], λ, αs_L1[k], ϵ)[0] for (k, λ) in enumerate(λs)]
plot_accuracy(acc_L1, i, "L1_Accuracy Plot", False)

#Part 3B
L1_w1, L1_w2, L1_w3 = L1_weights[:3]
pd.DataFrame(np.abs(L1_w1[1:]), column, columns=['Weights'] ).sort_values(by= "Weights", ascending=False).head()
pd.DataFrame(np.abs(L1_w2[1:]), column, columns=['Weights'] ).sort_values(by= "Weights", ascending=False).head()
pd.DataFrame(np.abs(L1_w3[1:]), column, columns=['Weights'] ).sort_values(by= "Weights", ascending=False).head()

#Part 3C
plot_sparsity(L1_weights, i, "L1_Sparsity plot")