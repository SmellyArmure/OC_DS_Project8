# # to run locally at test time
# time spark-submit --packages com.amazonaws:aws-java-sdk-pom:1.10.34,org.apache.hadoop:hadoop-aws:2.7.2,databricks:spark-deep-learning:1.5.0-spark2.4-s_2.11 P8_script_loc.py 

import os

# import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

## Configurations...
import os
path_cred = os.path.join(os.getcwd(),
            "AWS/AWS_IAM_CREDENTIAL/Maryse_P8_credentials.csv")
with open(path_cred,'r') as f:
        msg = f.read()
ID = str(msg).split('\n')[1].split(',')[2]
KEY = str(msg).split('\n')[1].split(',')[3]

## Define the details of the SparkSession
# spark = SparkSession.builder.appName('FeatExtraction').getOrCreate()
spark = SparkSession.builder.appName('FeatExtraction')\
                .config("spark.sql.shuffle.partitions",12)\
                .getOrCreate()
                # .config("spark.executor.memory", "2g")\

sc = spark.sparkContext.getOrCreate()

hadoopConf = sc._jsc.hadoopConfiguration()
hadoopConf.set("fs.s3a.access.key", ID)
hadoopConf.set("fs.s3a.secret.key", KEY)

## Create a Spark DataFrame containing all the pictures
import sparkdl

### Read images and vectorize
from pyspark.ml.image import ImageSchema

PREFIX = 'SAMPLE'

# Option1: Get local data
data_path = os.path.join("./DATA/fruits-360", PREFIX)
# Option2: Get data from s3
# bucket='ocfruitpictures'
# n_dir='data'
# data_path = 's3a://{}/{}/{}'.format(bucket, n_dir, PREFIX)
images_df = ImageSchema.readImages(data_path, recursive=True).repartition(12)

## Features extraction (Transfer Learning) using Sparkdl

from sparkdl import DeepImageFeaturizer

feat = DeepImageFeaturizer(inputCol="image",
                           outputCol="image_features",
                           modelName="ResNet50")
# ext_features_df = feat.transform(images_df)

## PCA on the extracted features

from pyspark.ml.feature import PCA

pca = PCA(k=8,
          inputCol="image_features",
          outputCol="pca_features")
# model = pca.fit(ext_features_df.select('image_features'))
# pca_feat_df = model.transform(ext_features_df)

## Put feature extractor and PCA in a pipeline

from pyspark.ml import Pipeline

pipe = Pipeline(stages=[feat, pca])
extractor = pipe.fit(images_df)
pca_feat_df = extractor.transform(images_df)

## Get the class of each image

import pyspark.sql.functions as pspfunc

orig_col = pca_feat_df['image']['origin']
split_col = pspfunc.split(orig_col, PREFIX+'/')
df_ = pca_feat_df.withColumn('labels', split_col.getItem(1))
split_col = pspfunc.split(df_['labels'], '/')
df_ = df_.withColumn('labels', split_col.getItem(0))
df_ = df_.withColumnRenamed("image", "path")

results_df = df_.select('path','pca_features','labels')
# results_df.show()
# store to local
path_res = "./RESULTS"
# store to s3
# path_res = "s3a://ocfruitpictures/results/res1"
results_df.write.mode('overwrite').parquet(path_res)

# input("Press Ctrl + C to escape !")
