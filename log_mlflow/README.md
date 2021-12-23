## HOWTO localhost

    mlflow ui
    # open http://localhost:5000

    python 01_first.py


## HOWTO

    # mlflow run
    export MLFLOW_S3_ENDPOINT_URL=http://10.205.126.222:30900
    export AWS_ACCESS_KEY_ID=P9SEFTER10OJ1YZLVQDN
    export AWS_SECRET_ACCESS_KEY=qOV44ONvN6+vnf6O+RId0f7hTqYyY+Ssr2uaYOoC
    mlflow server --backend-store-uri sqlite:///mydb.sqlite --default-artifact-root s3://rosbag

    # execute python script
    export MLFLOW_S3_ENDPOINT_URL=http://10.205.126.222:30900
    export AWS_ACCESS_KEY_ID=P9SEFTER10OJ1YZLVQDN
    export AWS_SECRET_ACCESS_KEY=qOV44ONvN6+vnf6O+RId0f7hTqYyY+Ssr2uaYOoC
    python 01_first.py

