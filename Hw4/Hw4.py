from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import time
import pyspark.sql.functions as F
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC, DecisionTreeClassifier
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import OneHotEncoder, StringIndexer, StandardScaler, VectorAssembler, SQLTransformer, IndexToString, VectorIndexer
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline, PipelineModel


# the initial year for training data
init_year = 2000

# the label column
target_col = 'Cancelled'
# the feature column
cate_cols = ['Month', 'DayofMonth', 'DayOfWeek', 'CRSDepTime', 'CRSArrTime', 'UniqueCarrier', 'FlightNum', 'CRSElapsedTime', 'Origin', 'Dest', 'Cancelled']
# the list of the model method: logistic regression, svm, random forest, decision tree
Methodlist = ['lr', 'svm', 'rf', 'dt']
# the used algorithm
method = Methodlist[2]

# create spark
spark = SparkSession.builder \
          .master("spark://node01-V1:7077")\
          .appName('node03.wusushi.{}'.format(method))\
          .config("spark.executor.memory", "2g") \
          .config("spark.cores.max", "6") \
          .getOrCreate()

sc = spark.sparkContext

# show dataframe
def dfShow(df):
	df.show()
	df.printSchema()
	df.count()

# load the file and filter the column
def dataLoad(file):
	# the feature column
	cate_cols = ['Month', 'DayofMonth', 'DayOfWeek', 'CRSDepTime', 'CRSArrTime', 'UniqueCarrier', 'FlightNum', 
					'CRSElapsedTime', 'Origin', 'Dest', 'Cancelled']
	df = spark.read.csv(file, header=True, nullValue='NA')
	# dfShow()

	df = df.select(cate_cols)
	df = df.withColumn('label', df[target_col].cast('float'))
	df = df.filter(df['label'].isNotNull())
	df = df.drop('Cancelled')
	cate_cols.pop()
	for col in cate_cols:
		df = df.filter(df[col].isNotNull())
	return df

# do undersampling to the data 
def UnderSampling(df):
	major_df = df.filter(F.col("label") == 0)
	minor_df = df.filter(F.col("label") == 1)
	print(major_df.count(), minor_df.count())
	ratio = int(major_df.count()/minor_df.count())
	print("ratio: {}".format(ratio))
	sampled_majority_df = major_df.sample(False, 1/ratio)
	combined_df_2 = sampled_majority_df.unionAll(minor_df)
	#dfShow(combined_df_2)
	return combined_df_2

# preprocess the training data and train the model
def dataPreprocessed(df):
	indexers = []
	encoders = []
	vectorinput = []

	for col in cate_cols:
		indexers.append(StringIndexer(inputCol=col, outputCol="{}_idx".format(col), handleInvalid='skip'))
		encoders.append(OneHotEncoder(inputCol="{}_idx".format(col), outputCol="{}_oh".format(col)))
		vectorinput.append("{}_oh".format(col))
	assembler = VectorAssembler(inputCols = vectorinput, outputCol = "_features", handleInvalid='skip')
	scaler = StandardScaler(inputCol='_features', outputCol='features', withStd=True, withMean=False)

	# select model
	if method == "lr":
		model = LogisticRegression(featuresCol='features', maxIter=5, regParam=0.1)
	elif method == "svm":
		model = LinearSVC(maxIter=15)
	elif method == "rf":
		model = RandomForestClassifier(numTrees=3)
	elif method == "dt":
		model = DecisionTreeClassifier()

	preprocessor = Pipeline(stages = indexers + encoders + [assembler, scaler] + [model]).fit(df)
	df = preprocessor.transform(df) 
	df.printSchema()
	return preprocessor

# create the model by training data and store the model
def modelTraining():
	path = "hdfs:///user/ubuntu/"
	paths = []
	for i in range(5):
	  paths.append("{}{}.csv".format(path, (init_year + i)))

	df = dataLoad(paths)
	combined_df_2 = UnderSampling(df)
	preprocessor = dataPreprocessed(combined_df_2)
	print('suss')

	# store model
	preprocessor.write().overwrite().save("{}/wusushi/model_{}".format(path, method))
	print("Save: {}/wusushi/model_{}".format(path, method))
	return preprocessor, combined_df_2

# evaluate the precision of the model
def dataEvaluation(test_df, path, flag):
	print("Load: {}/wusushi/model_{}".format(path, method))
	Model = PipelineModel.load("{}/wusushi/model_{}".format(path, method))
	predictions = Model.transform(test_df)
	preds_and_labels = predictions.select(['prediction', 'label'])

	metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
	precision = metrics.confusionMatrix().toArray()
	# inspect the data source
	if flag:
		print("Validation")
	else:
		print("Testting Data")

	# confusion matrix
	print(precision)
	print("TP: {}\tFP: {}".format(metrics.truePositiveRate(0.0), metrics.truePositiveRate(1.0)))
	print("Precision(0): {}\tPrecision(1): {}".format(metrics.precision(label=0.0), metrics.precision(label=1.0)))
	print("Recall(0): {}\tRecall(1): {}".format(metrics.recall(label=0.0), metrics.recall(label=1.0)))
	print("F-score(0): {}\tF-score(1): {}".format(metrics.fMeasure(label=0.0), metrics.fMeasure(label=1.0)))
	print("Accuracy: {}".format(metrics.accuracy))
	

def main():
	# train the model
	path = "hdfs:///user/ubuntu/"
	preprocessor, train_df = modelTraining()

	# evaluate training data
	train_df = train_df.randomSplit([0.85, 0.15], 0)[1]
	dataEvaluation(train_df, path, 1)

	# evaluate testing data
	test_df = dataLoad("{}2005.csv".format(path))
	dataEvaluation(test_df, path, 0)

if __name__ == '__main__':
	main()
