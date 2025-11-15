"""Training pipeline for confidence-aware classification experiments."""

import argparse

import torch,os,pandas as pd,datetime,numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm,time,joblib
from sklearn.preprocessing import StandardScaler, LabelBinarizer, MinMaxScaler

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, confusion_matrix)

from Losses import getLoss
from DataLoader import getData
from Model import MLP2Head


num_classes = 2
output_dim = num_classes

ModelDirectory="Models/"
os.makedirs(ModelDirectory,exist_ok=True)

def Train(epochs,torch_model,train_loader,optimizer,loss_fn,EXP):
    """Train the provided model for the requested number of epochs.

    Args:
        epochs (int): Number of passes over the training data.
        torch_model (torch.nn.Module): Neural network to optimize.
        train_loader (torch.utils.data.DataLoader): Batches of training data.
        optimizer (torch.optim.Optimizer): Optimizer instance used for updates.
        loss_fn (callable): Loss function compatible with the model outputs.
        EXP (str): Experiment identifier used when persisting artefacts.
    """
    # Train
    torch_model.train()
    pbar=tqdm.tqdm(range(epochs))
    for epoch in pbar:
        Ytrue,Ypred,Loss=[],[],[]
        for xb, yb in train_loader:
            xb,yb=xb.to(device),yb.to(device)
            pred = torch_model(xb)
            if isinstance(loss_fn,nn.MSELoss):
                yb=nn.functional.one_hot(yb,num_classes=2).to(torch.float32)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Loss.append(loss.item())
            if isinstance(pred,tuple):
                Ypred.append(pred[0])
            else:
                Ypred.append(pred)
            Ytrue.append(yb)
        Ytrue, Ypred=torch.concat(Ytrue,dim=0),torch.concat(Ypred,dim=0)
        if Ypred.ndim==1 or Ypred.shape[1] == 1:
            Ypred = (Ypred.ravel() >= 0.5).to(torch.int8)
        else:
            Ypred = torch.argmax(Ypred, dim=1)
        loss=torch.mean(torch.Tensor(Loss)).item()
        pbar.set_description(f"\nEpoch {epoch} Loss {loss} Accuracy {torch.sum(Ypred==Ytrue)/Ytrue.size()[0]}")
        # print(Ytrue.shape,Ypred.shape)
    ModelPath=ModelDirectory+f"Model_{EXP}.pt"
    torch.save(torch_model.state_dict(),ModelPath)

def Evaluate(torch_model, X_test_tensor, y_test, lb, dname="Test"):
    """Evaluate a trained model and compute summary metrics.

    Args:
        torch_model (torch.nn.Module): Model to evaluate.
        X_test_tensor (torch.Tensor): Feature tensor for evaluation split.
        y_test (torch.Tensor | np.ndarray | list): Ground-truth labels.
        lb (sklearn.preprocessing.LabelBinarizer): Fitted binarizer for ROC AUC.
        dname (str, optional): Name used to prefix metric keys. Defaults to "Test".

    Returns:
        dict: Mapping of metric names to their computed values for the split.
    """
    st = time.time()

    y_test_bin = lb.transform(y_test)
    torch_model.eval()

    with torch.no_grad():
        logits = torch_model(X_test_tensor.to(device))
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.cpu()

        # Handle 1-column or 2-column output
        if logits.ndim == 1 or logits.shape[1] == 1:
            # Binary classification (assume 2 classes)
            probs = logits.numpy().ravel()
            y_score = np.vstack([1 - probs, probs]).T
            y_pred = (probs >= 0.5).astype(int)
        else:
            # Multi-class classification
            y_score = torch.softmax(logits, dim=1).numpy()
            y_pred = np.argmax(y_score, axis=1)


    # ROC AUC
    try:
        if y_score.shape[1] > 2:
            roc_auc = roc_auc_score(y_test_bin, y_score, multi_class='ovr')
        else:
            roc_auc = roc_auc_score(y_test, y_score[:, 1])
    except:
        roc_auc = 0.0

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Best F1 search over thresholds (binary only)
    best_f1, best_thresh = 0, 0.5
    if y_score.shape[1] == 2:
        for t in np.arange(0.1, 0.9, 0.01):
            preds = (y_score[:, 1] >= t).astype(int)
            cf1 = f1_score(y_test, preds, average='macro', zero_division=0)
            if cf1 > best_f1:
                best_f1, best_thresh = cf1, t
    else:
        cf1 = f1  # Use normal F1 for multiclass

    results = {
        dname + "_Accuracy": acc,
        dname + "_Precision": prec,
        dname + "_Recall": rec,
        dname + "_F1 Score": cf1,
        dname + "_Best_F1 Score": best_f1,
        dname + "_ROC AUC": roc_auc,
        dname + "_Confusion Matrix": cm
    }

    print(f"{dname:10} acc={acc:<6.2f} prec={prec:<6.2f} rec={rec:<6.2f} f1={f1:<6.2f} "
          f"roc_auc={roc_auc:<8.2f} cm={cm} best_thresh={best_thresh:<8.2f} best_f1={best_f1:<8.2f}")

    print("Model Evaluated in ", time.time() - st, "seconds")
    return results


outFile=f"Results/Results.csv"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")



lossparams=dict(alpha_pos=1.0, alpha_neg=2.0, gamma=2.0, tau=0.7, lambda_cp=0.3, mu_ece=0.1, n_bins=15)
def Experiment(ModelName,DataType,SmotType,Scaller,ClassWeights,lossName,nlayers, hidden_dim,dropout_rate,use_batchnorm,LR,epochs,confcolumn,datapath,lossparams=lossparams):
    """Run a single end-to-end experiment and persist checkpoints and metrics.

    Args:
        ModelName (str): Registry key of the neural architecture to instantiate.
        DataType (str): Name of the dataset configuration to load.
        SmotType (str | None): Resampling strategy identifier or None.
        Scaller (str | None): Feature scaling strategy or None for raw values.
        ClassWeights (bool): Whether to compute class weights for the loss.
        lossName (str): Identifier of the loss implementation to use.
        nlayers (int): Number of hidden layers for the encoder.
        hidden_dim (int): Width of hidden layers.
        dropout_rate (float): Dropout probability applied after activations.
        use_batchnorm (bool): Whether to include batch-normalisation layers.
        LR (float): Learning rate for the optimizer.
        epochs (int): Number of optimizer steps over the dataset.
        confcolumn (str): Column storing the upstream confidence signal.
        datapath (str): Root directory containing the dataset files.
        lossparams (dict): Keyword arguments forwarded to `getLoss`.
    """
    EXP=datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    print(f"Running for Experiment {EXP} Model {ModelName} Loss {lossName} DataType {DataType}, Smote {SmotType} Scaller {Scaller} ")
    _,train_df,test_df=getData(DataType,SmotType,basepath=datapath)
    os.makedirs("Results",exist_ok=True)

    # Prepare features and labels
    X_train = train_df.drop(columns=["Label"])
    y_train = train_df["Label"]
    X_test = test_df.drop(columns=["Label"])
    y_test = test_df["Label"]
    X_Columns=list(X_train.columns)

    input_dim = X_train.shape[1]

    print("Train size",DataType,X_train.shape,"Test Size",X_test.shape)
    print("Train Labels",np.unique(y_train,return_counts=True),"Test Size",np.unique(y_test,return_counts=True))
    if Scaller is not None:
        # Scale features
        if Scaller=="standard":
            scaler = StandardScaler()
        if Scaller=="minmax":
            scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train.values
        X_test = X_test.values

    # Label binarizer for ROC AUC (handles multiclass)
    lb = LabelBinarizer()
    lb.fit(y_train)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)

    loss_fn, ClassWeights, lossName = getLoss(lossName,ClassWeights,y_train,device,SmotType,**lossparams)

    if ModelName=="MLP2HEAD":
        confidencecolumnIndex=X_Columns.index(confcolumn)
        torch_model = MLP2Head(input_dim, nlayers, hidden_dim, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm,confidencecolumnIndex=confidencecolumnIndex).to(device)

    optimizer = optim.Adam(torch_model.parameters(), lr=LR)
    st=time.time()
    Train(epochs,torch_model,train_loader,optimizer,loss_fn,EXP)
    if Scaller is not None:
        ScallerPath = ModelDirectory + f"Scaller_{EXP}"
        joblib.dump(scaler, ScallerPath)

    print("Model trained in ",time.time()-st,"seconds")

    header = f"{'Mode':<10} {'Acc':<10} {'Prec':<10} {'Rec':<10} {'F1':<10} {'ROC AUC':<10}"
    print(header)
    print("-" * (len(header)*2))

    currentResults=dict(
        EXP=EXP,
        Model=ModelName,lossName=lossName, confcolumn=confcolumn,
        Nlayers=nlayers, hidden_dim=hidden_dim,epochs=epochs,Learningrate=LR,
        dropout_rate=dropout_rate,use_batchnorm=use_batchnorm,
        Smote=SmotType,
        ClassWeights=ClassWeights,
        Scaller=Scaller,
        Data=DataType,
        NumberofFeatures=input_dim,
        TotalTrain=X_train.shape[0],
        TotalTest=X_test.shape[0],
        Lossargs=str(loss_fn),
        **Evaluate(torch_model,X_train_tensor,y_train_tensor,lb,dname="Train"),
        **Evaluate(torch_model, X_test_tensor, y_test_tensor,lb, dname="Test")
    )


    if os.path.exists(outFile):
        df = pd.read_csv(outFile)
        results = [row.to_dict() for i, row in df.iterrows()]
    else:
        results = []
    results.append(currentResults)
    # Save all results
    results_df = pd.DataFrame(results)
    results_df.to_csv(outFile, index=False)

    print("✅ All evaluations complete. Results saved to",outFile)


def get_arg_parser():
    """Build the argument parser describing supported experiment knobs.

    Returns:
        argparse.ArgumentParser: Parser configured with CLI options.
    """
    parser = argparse.ArgumentParser(description="Deep Learning Experiment with ArgParser")

    # Model and dataset
    parser.add_argument("--datapath", default="Dataset")
    parser.add_argument("--Model", default="MLP2HEAD")
    parser.add_argument("--Data", default="stat_self_critique", choices=["stat_self_critique"])
    parser.add_argument("--lossName", default="FocusCalLoss")

    # Training configuration
    parser.add_argument("--Nlayers", default=5, type=int, choices=[2, 3, 4, 5])
    parser.add_argument("--hidden_dim", default=200, type=int, choices=[200, 500, 1000])
    parser.add_argument("--dropout_rate", default=0.2, type=float)
    parser.add_argument("--use_batchnorm", default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument("--Learningrate", default=0.001, type=float)
    parser.add_argument("--epochs", default=50, type=int)

    # Data balancing and scaling
    parser.add_argument("--Sampler", default="none", choices=["SMOTE", "RandomUnderSampler", "none"])
    parser.add_argument("--Scaller", default="standard", choices=["standard", "minmax", "none"])
    parser.add_argument("--ClassWeights", default=False, type=lambda x: x.lower() in ['true', '1'])

    # Loss parameters
    parser.add_argument("--alpha_pos", default=2.0, type=float, choices=[2, 3, 4])
    parser.add_argument("--alpha_neg", default=1.0, type=float, choices=[1, 2, 3])
    parser.add_argument("--gamma", default=2, type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--lambda_cp", default=6, type=int, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--mu_ece", default=5, type=int, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--tau", default=0.7, type=float)
    parser.add_argument("--n_bins", default=15, type=int)
    parser.add_argument("--confcolumn", default="InitialConfidence")

    return parser

def Main():
    """Entry point for command-line execution."""
    parser = get_arg_parser()
    args = parser.parse_args()

    # Handle "none" -> None conversion
    args.Smote = None if args.Sampler == "none" else args.Sampler
    args.Scaller = None if args.Scaller == "none" else args.Scaller

    # Build loss parameter dictionary (same as in your DeepLearningGridSearch)
    lossparam = {
        "alpha_pos": args.alpha_pos,"alpha_neg": args.alpha_neg,"gamma": args.gamma,"tau": args.tau,
        "lambda_cp": args.lambda_cp,"mu_ece": args.mu_ece,"n_bins": args.n_bins,
    }

    # Call the experiment
    Experiment(ModelName=args.Model,DataType=args.Data,SmotType=args.Smote,Scaller=args.Scaller,ClassWeights=args.ClassWeights,lossName=args.lossName,nlayers=args.Nlayers,hidden_dim=args.hidden_dim,dropout_rate=args.dropout_rate,use_batchnorm=args.use_batchnorm,LR=args.Learningrate,epochs=args.epochs,confcolumn=args.confcolumn,datapath=args.datapath,lossparams=lossparam)

if __name__ == "__main__":
    Main()