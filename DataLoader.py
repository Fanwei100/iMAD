import os.path,time
from functools import reduce
from imblearn.over_sampling import SMOTE,SMOTEN,SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
import pandas as pd

DataSets=["GQA", "PUBMEDQA"]

BASEPATH="dataset/" # This is the training dataset for classifier


def ReadExcelFiles(filpath):
	"""Read every sheet from an Excel workbook into a dictionary.

	Args:
		filpath (str): Absolute path to the Excel file.

	Returns:
		dict[str, pandas.DataFrame]: Mapping of sheet names to data frames.
	"""
	return pd.read_excel(filpath, sheet_name=None, dtype={"ID": str})

def convert_to_float(x):
	"""Convert values that might be lists or strings into floats.

	Args:
		x (Any): Value loaded from Excel.

	Returns:
		float: Normalised floating-point representation of ``x``.
	"""
	if isinstance(x, list) and len(x) == 1:
		return float(x[0])
	elif isinstance(x,str):
		if "[" in x: x = x.replace("[","")
		if "]" in x: x = x.replace("]","")
	return float(x)  # Already float or coercible

def LoadDataset(FeatureFiles,LabelFile,SmotType=None,datasets = DataSets,requiremetadata=False):
	"""Load feature and label workbooks, merge splits, and optionally resample.

	Args:
		FeatureFiles (list[str] | str): Paths to feature workbooks.
		LabelFile (str): Path to the labels workbook.
		SmotType (str | None): Name of the resampler to apply or ``None``.
		datasets (list[str]): Sheet names representing datasets of interest.
		requiremetadata (bool): Keep metadata columns when ``True``.

	Returns:
		tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, ...]:
			Full dataset, stratified training DataFrame, and evaluation DataFrame.
			If ``requiremetadata`` is True, additional metadata splits are included.
	"""
	st = time.time()
	if isinstance(FeatureFiles, str):
		FeatureFiles=[FeatureFiles]
	FeatureDictList=[]
	for ff in FeatureFiles:
		dfdict=ReadExcelFiles(ff)
		FeatureDictList.append(dfdict)
	LabelsDict = pd.read_excel(LabelFile, sheet_name=None, dtype={"ID": str})
	dflist = []
	for data in datasets:
		if data not in LabelsDict:
			print(data,"Not Present Skipping...")
			continue
		labels = LabelsDict[data]
		DataList = [FeatureDict[data] for FeatureDict in FeatureDictList] + [labels]
		df = reduce(lambda left, right: pd.merge(left, right, on='ID', how='inner'), DataList)
		if requiremetadata:
			df["Data"]=data
		else:
			df=df.drop(columns=["ID"])
		dflist.append(df)
	FullData = pd.concat(dflist, axis=0)
	# Separate features (X) and target (y)
	X = FullData.drop('Label', axis=1)
	Y = FullData['Label']
	FeatureColumns=X.columns
	if requiremetadata: FeatureColumns=list(filter(lambda x:x not in ["Data","ID"],FeatureColumns))
	for col in FeatureColumns:
		X[col]= X[col].apply(convert_to_float)
	# Split data if needed (optional but recommended)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, random_state=42, test_size=0.2)
	if SmotType is not None:
		assert requiremetadata is False, "Can't do smote with metadata"
		# Apply SMOTE
		smote={"SMOTE":SMOTE,"SMOTEN":SMOTEN,"SMOTENC":SMOTENC,"SMOTETomek":SMOTETomek,"RandomUnderSampler":RandomUnderSampler}[SmotType](random_state=42)
		X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
	else:
		X_resampled, y_resampled = X_train, y_train
	# Convert back to DataFrame if needed
	X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
	y_resampled_df = pd.Series(y_resampled, name='Label')
	# Combine back into a single DataFrame (optional)
	traindf = pd.concat([X_resampled_df, y_resampled_df], axis=1)

	featuredf = pd.DataFrame(X_test, columns=X.columns)
	Labeldf = pd.Series(y_test, name='Label')
	testdf = pd.concat([featuredf, Labeldf], axis=1)
	print("Data Loaded in ", time.time() - st, "seconds")
	traindf=traindf.reset_index(drop=True)
	testdf=testdf.reset_index(drop=True)
	if not requiremetadata:
		return  FullData,traindf,testdf
	trainDataInfo=traindf[["Data","ID"]]
	testDataInfo=testdf[["Data","ID"]]
	traindf.pop("Data")
	traindf.pop("ID")
	testdf.pop("Data")
	testdf.pop("ID")
	return FullData,traindf,testdf,(trainDataInfo,testDataInfo)


def getData(DataType,SmotType=None,datasets=DataSets,requiremetadata=False,basepath=BASEPATH):
	"""Load the configured dataset and return train/test partitions.

	Args:
		DataType (str): Name of the dataset preset to load.
		SmotType (str | None): Resampling strategy identifier or ``None``.
		datasets (list[str]): Which dataset sheets to include.
		requiremetadata (bool): Include metadata columns when True.
		basepath (str): Directory containing dataset Excel files.

	Returns:
		tuple: Composite output from ``LoadDataset`` tailored to the request.
	"""
	assert DataType=="self_critique"
	if basepath[-1]!="/":
		basepath=basepath+"/"
	ALLData= LoadDataset(basepath+"self_critique_features.xlsx",basepath+"labels.xlsx",SmotType=SmotType,datasets=datasets,requiremetadata=requiremetadata)
	train_df,test_df=ALLData[1],ALLData[2]
	if len(ALLData)==3:
		return  ALLData[0],train_df,test_df
	else:
		return ALLData[0], train_df, test_df,*ALLData[3:]










