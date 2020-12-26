from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import time
from operator import add
import re


#local worker
spark = SparkSession.builder \
			.master("local[*]")\
			.appName('user')\
			.getOrCreate()

# #yarn cluster
# spark = SparkSession.builder \
# 			.master("yarn")\
# 			.appName('user')\
# 			.getOrCreate()

sc = spark.sparkContext

# count the sentences in this article
def CountSentences(textfile):
	sentences = 0
	flag = True   # the flag is true until the sentences read '“', and the the flag is false, otherwise, the flag is false until the sentences read '”'

	for i in range(len(textfile)):
		for j in range(len(textfile.loc[i][0])):
			if textfile.loc[i][0][j] == '“':
				flag = False
			elif textfile.loc[i][0][j] == '”':
				flag = True
				if j + 1 < len(textfile.loc[i][0]):
					if textfile.loc[i][0][j + 1] >= 'A' and textfile.loc[i][0][j + 1] <= 'Z':
						# if the '”' is the end in this sentences, the sentence counts plus 1
						sentences += 1
			elif (textfile.loc[i][0][j] == '.' or textfile.loc[i][0][j] == '?') and flag:
				sentences += 1
	return sentences

def StringProcessing(string):
	string = re.split(r'[\s+,.“”?—]+', string)   # split the sentences
	return string

def Q1CalculateWord():
	a = time.time()                               # record the current time
	url = 'hdfs:///user/ubuntu/Youvegottofindwhatyoulove.txt'
	textfile = pd.read_table(url, header=None)    # read the file by pandas type 
	sentences = CountSentences(textfile)          # count the numbers of sentences
	rdd = sc.parallelize(textfile.loc[:, 0])      # parallelize the data
	counts = (rdd.flatMap(lambda line: StringProcessing(line))     # word count
		.map(lambda x: (x, 1))
		.reduceByKey(add)
		.collect()
	)
	counts = sorted(counts, key=lambda x: x[1], reverse=True)      # sort the counts of the words
	for i in range(30):
		print(counts[i], format(counts[i][1]/sentences, '.2f'))    # show the top 30 words
	print("Total: {} seconds".format(time.time() - a))             # show the execution time

# calculate the numbers of the payment type by credit card 
def Paymenttype(rows):
	if rows[1] == "1" and int(rows[0]) >= 1 and int(rows[0]) <= 4:
		return (rows[0], 1)
	else:
		return ("0", 0)

# calculate the numbers of the different number of passengers in credit card trip
def DataProcessing(rows):
	if rows[1] == "1":
		if int(rows[0]) >= 1 and int(rows[0]) <= 4:
			return (rows[0], rows[2])
		else:      # the numbers of the passengers out of domain
			return ("0", 0)
	else:          # missing data
		return ("0", 0)

def Q2calculateamount():             
	a = time.time()                   # record the current time
	url = 'hdfs:///user/ubuntu/yellow_tripdata_2017-09.csv'
	taxifile = pd.read_csv(url,       # read the file by pandas type and read the specialized columns which I have to use
				usecols=["passenger_count", "payment_type", "total_amount"], 
				dtype={"passenger_count" : "str", "payment_type" : "str", "total_amount" : "float"}
			)
	location = ["passenger_count", "payment_type", "total_amount"]
	rdd = sc.parallelize(taxifile.loc[:, location].values.tolist())       # parallelize the data
	cardcounts = (rdd.map(lambda rows: Paymenttype(rows))     # counts the numbers which is in credit card trip
		.reduceByKey(add)
		.collect()
	)
	counts = (rdd.map(lambda rows: DataProcessing(rows))      # calculate the amount in credit card trip
		.reduceByKey(add)
		.collect()
	)
	for i in range(len(counts)):
		print(counts[i][0], counts[i][1]/cardcounts[i][1])    # show the data
	print("Total: {} seconds".format(time.time() - a))        # show the execution time

def main():
	Q1CalculateWord()      # the first question
	Q2calculateamount()    # the second question

if __name__ == '__main__':
	main()