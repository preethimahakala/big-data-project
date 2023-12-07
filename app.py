import streamlit as st
import logging
import pandas as pd
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScalerModel
from pyspark.ml.feature import PCAModel
from pyspark.ml.classification import GBTClassificationModel

spark = SparkSession.builder.appName("Project")\
    .config("spark.driver.memory", "15g")\
    .config("spark.executor.memory", "15g")\
    .config("spark.memory.offHeap.enabled",True)\
    .config("spark.memory.offHeap.size","15g")\
    .getOrCreate()

with open("kddcup.names", "r") as f:
    names = f.readlines()
    dtypes = [name.split(":")[1].strip()[:-1] for name in names[1:]]
    dtypes = ['float' if dtype == 'continuous' else 'string' for dtype in dtypes]
    names = [name.split(":")[0] for name in names[1:]]

schema = StructType()
for name, dtype in zip(names, dtypes):
    schema.add(name, dtype)
dtype_dict = dict(zip(names, dtypes))

pre_pipeline = PipelineModel.load("pre_pipeline")
scaler = StandardScalerModel.load("scalerModel")
pca = PCAModel.load("pcaModel")
model = GBTClassificationModel.load("best_model")

def predict(file):
    if file:
        pdf = pd.read_csv(file, header=None, names=names)
        for col in pdf.columns:
            if dtype_dict[col] == 'float':
                pdf[col] = pdf[col].astype(float)
        df = spark.createDataFrame(pdf, schema=schema)
    else:
        return None
    df = pre_pipeline.transform(df)
    df = df.drop("protocol_type", "land", "logged_in", "is_host_login", "is_guest_login", "service", "flag", "protocol_type_index", "land_index", "logged_in_index", "is_host_login_index", "is_guest_login_index", "service_index", "flag_index")
    # print a dictionary of the columns and their data types
    print({col.name: col.dataType for col in df.schema})
    assembler = VectorAssembler(inputCols=df.columns, outputCol="features")
    df = assembler.transform(df).select("features")
    df = scaler.transform(df).select("scaled_features")
    df = pca.transform(df).select("pca_features")
    df = model.transform(df).select("prediction")
    return df.collect()

def app():
    st.title("Intrusion Detection System")

    st.write("This is a simple IDS based on the KDD Cup 1999 dataset. It uses a Gradient Boosted Tree classifier to classify network traffic into normal or attack.")

    # put a file uploader and a submit button
    # a prediction is a spark dataframe with one column: prediction and might have multiple rows
    file = st.file_uploader("Upload a csv file", type=["csv"])
    if file:
        st.write("File uploaded")
    if st.button("Submit"):
        predictions = predict(file)
        if predictions:
            st.write("Predictions:")
            st.write(predictions)
        else:
            st.write("No file uploaded")


if __name__ == "__main__":
    app()